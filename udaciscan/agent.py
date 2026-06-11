"""
agent.py — Main orchestration for the UdaciScan drug repurposing agent.

Implements a stateful 7-step workflow:
  1. retrieve        : query ChromaDB for top-k relevant docs
  2. evaluate        : compute confidence score
  3. live_fetch      : fallback to PubMed if confidence < τ (with retry)
  4. requery         : re-retrieve after upsert
  5. rerank          : LLM-based relevance reranking
  6. extract & score : LLM candidate extraction + deterministic scoring
  7. synthesize      : build RepurposingBrief, save JSON + Markdown, log trace

Usage (CLI):
    python agent.py --disease "Parkinson's disease"
    python agent.py --disease "NASH" --k 15 --tau 0.60
    python agent.py --diseases "Parkinson's disease" "NASH" "Glioblastoma"
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env"))

from config import load_settings, Settings
from schema import CandidateDrug, RepurposingBrief
from scoring import score_and_rank
from reporting import save_json, save_markdown, log_step, make_trace_path
from tooling import (
    tool_retrieve_internal,
    tool_evaluate_retrieval,
    tool_fetch_live,
    tool_rerank,
    tool_extract_candidates,
)

# ---------------------------------------------------------------------------
# Default output directories
# ---------------------------------------------------------------------------
_OUTPUTS_DIR = os.path.join(_ROOT, "outputs")
_RUNS_DIR    = os.path.join(_ROOT, "runs")

# ---------------------------------------------------------------------------
# Method summary template
# ---------------------------------------------------------------------------
_METHOD_SUMMARY = (
    "UdaciScan used a two-tier retrieval strategy: (1) primary RAG over a local "
    "ChromaDB collection seeded with PubMed abstracts; (2) live PubMed fallback "
    "when retrieval confidence fell below τ={tau:.2f}. Retrieved documents were "
    "reranked by an LLM, then drug candidates were extracted via structured LLM "
    "information extraction and scored deterministically using evidence tier, "
    "study model/outcome, mechanism coverage, evidence volume, and extractor "
    "confidence (weights from config.yaml). All cited PMIDs were validated "
    "against the retrieval cache."
)

_LIMITATIONS = (
    "Coverage is limited to PubMed abstracts available at query time; full-text "
    "content was not systematically used. Evidence tiers are inferred from abstract "
    "language and may not reflect the actual trial phase. Candidates with limited "
    "PubMed representation may be under-scored. This report is generated for "
    "research purposes only and does NOT constitute medical advice."
)


# ---------------------------------------------------------------------------
# Core agent workflow
# ---------------------------------------------------------------------------

def run_agent(
    disease: str,
    settings: Settings,
    k: Optional[int] = None,
    tau: Optional[float] = None,
    output_dir: str = _OUTPUTS_DIR,
    runs_dir: str = _RUNS_DIR,
    trace_path: Optional[str] = None,
) -> RepurposingBrief:
    """
    Run the full UdaciScan pipeline for a single disease.

    Args:
        disease:    Target disease name (e.g. "Parkinson's disease").
        settings:   Loaded Settings object.
        k:          Override top-k retrieval (defaults to settings.k).
        tau:        Override confidence threshold (defaults to settings.tau).
        output_dir: Directory for JSON/Markdown outputs.
        runs_dir:   Directory for JSONL trace files.
        trace_path: Reuse an existing trace file (useful for multi-disease runs).

    Returns:
        A validated RepurposingBrief instance.
    """
    k = k or settings.k
    tau = tau if tau is not None else settings.tau

    # Create trace file for this run if not provided
    if trace_path is None:
        trace_path = make_trace_path(runs_dir)

    print(f"\n{'='*60}")
    print(f"  UdaciScan | Disease: {disease}")
    print(f"  k={k} | τ={tau:.2f} | trace -> {os.path.basename(trace_path)}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Step 1 — RETRIEVE (internal)
    # ------------------------------------------------------------------
    print(f"[1/7] Retrieving top-{k} docs from ChromaDB...")
    docs = tool_retrieve_internal(disease, settings, k=k, trace_path=trace_path)
    print(f"      -> {len(docs)} docs retrieved")

    # ------------------------------------------------------------------
    # Step 2 — EVALUATE confidence
    # ------------------------------------------------------------------
    print(f"[2/7] Evaluating retrieval confidence (τ={tau:.2f})...")
    confidence = tool_evaluate_retrieval(disease, docs, settings, trace_path=trace_path)
    print(f"      -> confidence = {confidence:.4f} | fallback = {confidence < tau}")

    # ------------------------------------------------------------------
    # Step 3 — LIVE FETCH + UPSERT (fallback if confidence < τ)
    # ------------------------------------------------------------------
    attempts = 0
    while confidence < tau and attempts < settings.max_attempts:
        attempts += 1
        print(f"[3/7] Confidence below τ — live fetch attempt {attempts}/{settings.max_attempts}...")
        n_upserted = tool_fetch_live(disease, settings, trace_path=trace_path)
        print(f"      -> {n_upserted} new docs upserted into ChromaDB")

        # ------------------------------------------------------------------
        # Step 4 — RE-QUERY after upsert
        # ------------------------------------------------------------------
        print(f"[4/7] Re-querying ChromaDB after upsert...")
        docs = tool_retrieve_internal(disease, settings, k=k, trace_path=trace_path)
        confidence_after = tool_evaluate_retrieval(
            disease, docs, settings, trace_path=trace_path
        )

        log_step(
            trace_path,
            step="requery",
            params={"attempt": attempts, "tau": tau},
            result={
                "confidence_before": round(confidence, 4),
                "confidence_after":  round(confidence_after, 4),
                "n_docs":            len(docs),
                "improved":          confidence_after > confidence,
            },
        )

        print(f"      -> confidence after re-query: {confidence_after:.4f}")
        confidence = confidence_after

    if attempts == 0:
        # No fallback needed — log step 3/4 as skipped
        log_step(
            trace_path,
            step="requery",
            params={"tau": tau},
            result={"skipped": True, "reason": "confidence >= tau", "confidence": round(confidence, 4)},
        )
        print(f"[3-4/7] Fallback not needed (confidence >= τ)")

    # ------------------------------------------------------------------
    # Step 5 — RERANK
    # ------------------------------------------------------------------
    print(f"[5/7] Reranking {len(docs)} docs with LLM...")
    docs = tool_rerank(disease, docs, settings, k=k, trace_path=trace_path)
    print(f"      -> {len(docs)} docs after rerank")

    # ------------------------------------------------------------------
    # Step 6 — EXTRACT candidates + SCORE & RANK
    # ------------------------------------------------------------------
    print(f"[6/7] Extracting candidates with LLM...")
    candidates: List[CandidateDrug] = tool_extract_candidates(
        disease, docs, settings, trace_path=trace_path
    )
    print(f"      -> {len(candidates)} candidates extracted")

    if candidates:
        candidates = score_and_rank(candidates, settings)
        print(f"      -> candidates scored and ranked")

        log_step(
            trace_path,
            step="score",
            params={"disease": disease},
            result={
                "n_candidates": len(candidates),
                "top_drug":     candidates[0].drug_name if candidates else None,
                "top_score":    round(candidates[0].score, 4) if candidates else None,
                "candidates":   [c.model_dump() for c in candidates],
            },
        )
        
        # Compact candidates table
        TIER_SHORT = {"preclinical": "Preclin.", "phase-1": "Ph.1", "phase-2": "Ph.2", "phase-3": "Ph.3"}

        COL_RANK   = 4
        COL_DRUG   = 24
        COL_TIER   = 8
        COL_MODEL  = 8
        COL_OUT    = 8
        COL_SCORE  = 7
        COL_CONF   = 6
        COL_PMIDS  = 20

        header = (
            f"  {'#':<{COL_RANK}}  {'Drug':<{COL_DRUG}}  {'Tier':<{COL_TIER}}"
            f"  {'Model':<{COL_MODEL}}  {'Outcome':<{COL_OUT}}"
            f"  {'Score':>{COL_SCORE}}  {'Conf':>{COL_CONF}}  {'PMIDs':<{COL_PMIDS}}"
        )
        sep = (
            f"  {'─'*COL_RANK}  {'─'*COL_DRUG}  {'─'*COL_TIER}"
            f"  {'─'*COL_MODEL}  {'─'*COL_OUT}"
            f"  {'─'*COL_SCORE}  {'─'*COL_CONF}  {'─'*COL_PMIDS}"
        )
        print(f"\n{header}")
        print(sep)
        for rank, c in enumerate(candidates, start=1):
            tier   = TIER_SHORT.get(c.evidence_tier, c.evidence_tier)
            pmids  = ", ".join(c.pmids)[:COL_PMIDS]
            drug   = c.drug_name[:COL_DRUG]
            model  = c.model[:COL_MODEL]
            outcome = c.outcome[:COL_OUT]
            print(
                f"  {rank:<{COL_RANK}}  {drug:<{COL_DRUG}}  {tier:<{COL_TIER}}"
                f"  {model:<{COL_MODEL}}  {outcome:<{COL_OUT}}"
                f"  {c.score:>{COL_SCORE}.3f}  {c.confidence:>{COL_CONF}.2f}  {pmids:<{COL_PMIDS}}"
            )
        print()
        
    else:
        print(f"       No candidates extracted - brief will be empty")
        log_step(
            trace_path,
            step="score",
            params={"disease": disease},
            result={"n_candidates": 0},
        )

    # ------------------------------------------------------------------
    # Step 7 — SYNTHESIZE brief + save outputs
    # ------------------------------------------------------------------
    print(f"[7/7] Synthesizing brief and saving outputs...")

    brief = RepurposingBrief(
        disease        = disease,
        date           = date.today().isoformat(),
        method_summary = _METHOD_SUMMARY.format(tau=tau),
        candidates     = candidates,
        limitations    = _LIMITATIONS,
    )

    json_path = save_json(brief, output_dir=output_dir)
    md_path   = save_markdown(brief, output_dir=output_dir)

    log_step(
        trace_path,
        step="synthesize",
        params={"disease": disease},
        result={
            "n_candidates":  len(brief.candidates),
            "json_output":   os.path.relpath(json_path),
            "md_output":     os.path.relpath(md_path),
            "final_confidence": round(confidence, 4),
            "candidates":    [c.model_dump() for c in brief.candidates],
        },
    )

    print(f"\nBrief saved:")
    print(f"  JSON     -> {os.path.relpath(json_path)}")
    print(f"  Markdown -> {os.path.relpath(md_path)}")
    print(f"  Trace    -> {os.path.relpath(trace_path)}")

    return brief


# ---------------------------------------------------------------------------
# Multi-disease runner
# ---------------------------------------------------------------------------

def run_multi(
    diseases: List[str],
    settings: Settings,
    k: Optional[int] = None,
    tau: Optional[float] = None,
    output_dir: str = _OUTPUTS_DIR,
    runs_dir: str = _RUNS_DIR,
) -> List[RepurposingBrief]:
    """
    Run the agent sequentially for multiple diseases.
    Each disease gets its own trace file.

    Args:
        diseases:   List of disease names.
        settings:   Loaded Settings object.
        k:          Override top-k retrieval.
        tau:        Override confidence threshold.
        output_dir: Directory for JSON/Markdown outputs.
        runs_dir:   Directory for JSONL trace files.

    Returns:
        List of RepurposingBrief instances (one per disease).
    """
    briefs = []
    for disease in diseases:
        brief = run_agent(
            disease    = disease,
            settings   = settings,
            k          = k,
            tau        = tau,
            output_dir = output_dir,
            runs_dir   = runs_dir,
        )
        briefs.append(brief)
    return briefs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UdaciScan — AI Drug Repurposing Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python agent.py --disease \"Parkinson's disease\"\n"
            "  python agent.py --diseases \"Parkinson's disease\" \"NASH\" \"Glioblastoma\"\n"
            "  python agent.py --disease \"NASH\" --k 15 --tau 0.60\n"
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--disease", type=str,
        help="Single target disease name."
    )
    group.add_argument(
        "--diseases", nargs="+",
        help="One or more target disease names (space-separated)."
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override top-k retrieval (default: from config.yaml)."
    )
    parser.add_argument(
        "--tau", type=float, default=None,
        help="Override confidence threshold τ (default: from config.yaml)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=_OUTPUTS_DIR,
        help=f"Output directory for briefs (default: {_OUTPUTS_DIR})."
    )
    parser.add_argument(
        "--runs-dir", type=str, default=_RUNS_DIR,
        help=f"Directory for trace files (default: {_RUNS_DIR})."
    )
    parser.add_argument(
        "--config", type=str, default=os.path.join(_ROOT, "config.yaml"),
        help="Path to config.yaml."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Load settings (from config.yaml, with optional overrides)
    settings = load_settings(args.config)
    if args.tau is not None:
        settings.tau = args.tau

    diseases = [args.disease] if args.disease else args.diseases

    briefs = run_multi(
        diseases   = diseases,
        settings   = settings,
        k          = args.k,
        tau        = args.tau,
        output_dir = args.output_dir,
        runs_dir   = args.runs_dir,
    )

    print(f"\n{'='*60}")
    print(f"  Done. {len(briefs)} brief(s) produced.")
    print(f"{'='*60}\n")