"""
tooling.py — Agent tool wrappers for UdaciScan.

Exposes five callable tools used by agent.py:
  1. tool_retrieve_internal  : query ChromaDB for top-k relevant docs
  2. tool_evaluate_retrieval : compute a confidence score from retrieved docs
  3. tool_fetch_live         : pull fresh articles from PubMed, embed & upsert
  4. tool_rerank             : LLM-based reranking of retrieved docs
  5. tool_extract_candidates : LLM information extraction → List[CandidateDrug]

All tools accept a trace_path argument (optional) to log their step via reporting.py.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup — allow imports from the parent directory (retrieval_tools, etc.)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(_ROOT, ".env"))

from config import Settings
from retrieval_tools import retrieve_internal, evaluate_retrieval
from reporting import log_step
from schema import CandidateDrug

# ---------------------------------------------------------------------------
# OpenAI client (shared, lazy-initialised)
# ---------------------------------------------------------------------------

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url="https://openai.vocareum.com/v1")
    return _client


# ---------------------------------------------------------------------------
# Drug class / non-drug filter
# ---------------------------------------------------------------------------

# Suffixes that signal a pharmacological CLASS rather than a specific drug
_CLASS_SUFFIX_RE = re.compile(
    r"\b("
    r"inhibitors?|antagonists?|agonists?|modulators?|blockers?|"
    r"activators?|inducers?|sensitizers?|promoters?|suppressors?|"
    r"analogues?|analogs?|derivatives?|agents?|drugs?|compounds?|"
    r"therapies|therapy|treatments?|interventions?|approaches?|"
    r"protocols?|procedures?|devices?|platforms?|technologies?|"
    r"antibiotics?|antivirals?|antifungals?|antiparasitics?|"
    r"vaccines?|biologics?|gene\\s+therapies?"
    r")\s*$",
    re.IGNORECASE,
)

# Known broad class names (exact or near-exact match)
_CLASS_NAMES = {
    "statins", "nsaids", "ssris", "snris", "opioids", "opiates", "benzodiazepines", "corticosteroids",
    "glucocorticoids", "mineralocorticoids", "retinoids", "bisphosphonates", "dmards", "immunosuppressants",
    "antidepressants", "antipsychotics", "anticonvulsants", "antihypertensives", "diuretics", 
    "beta-blockers", "ace inhibitors", "nitrates", "fibrates", "monoclonal antibodies", "checkpoint inhibitors",
    "kinase inhibitors", "protease inhibitors", "nucleoside analogues", "nucleotide analogues",
    "insulin analogues", "cytokines", "interferons", "interleukins", "chemotherapy", "radiation", "surgery", 
    "exercise", "diet", "placebo", "vehicle", "control",
}

# Words that clearly indicate non-pharmacological entities
_NON_DRUG_WORDS_RE = re.compile(
    r"\b(device|implant|surgery|surgical|radiation|exercise|"
    r"vitamin|mineral|probiotic|crispr|sirna|shrna|mirna|antisense|oligonucleotide|"
    r"nanoparticle|transplant|transfusion|dialysis|phototherapy|immunotherapy\\s+platform)\b",
    re.IGNORECASE,
)

def _is_broad_class_or_non_drug(drug_name: str) -> bool:
    """
    Return True if drug_name looks like a broad class, device, or
    non-pharmacological intervention rather than a specific drug.

    Allowed: single substances , fixed-dose combos ("lopinavir/ritonavir"), 
    brand names ("Herceptin").
    """
    name = drug_name.strip().lower()

    # 1. Exact match against known class list
    if name in _CLASS_NAMES:
        return True

    # 2. Suffix signals a class ("ACE inhibitors", "mTOR inhibitors")
    if _CLASS_SUFFIX_RE.search(drug_name):
        return True

    # 3. Non-drug / non-pharmacological keywords
    if _NON_DRUG_WORDS_RE.search(drug_name):
        return True

    # 4. Long name with spaces -> likely a description, not a drug
    #    Fixed-dose combos like "lopinavir/ritonavir" contain "/" -> allowed
    words = name.split()
    if len(words) >= 4 and "/" not in name:
        return True

    return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_pmid_cache(docs: List[Dict[str, Any]]) -> set:
    """Collect all PMIDs present in the retrieved doc set."""
    cache = set()
    for d in docs:
        meta = d.get("metadata") or {}
        pmid = meta.get("pmid")
        if not pmid:
            doc_id = d.get("id", "")
            if isinstance(doc_id, str) and doc_id.startswith("PMID:"):
                pmid = doc_id.split("PMID:")[-1]
        if pmid:
            cache.add(str(pmid).strip())
    return cache


def _embed_text(text: str, settings: Settings) -> List[float]:
    """Embed a single string using the configured embedding model."""
    client = _get_client()
    resp = client.embeddings.create(model=settings.embed_model, input=[text])
    return resp.data[0].embedding


# ---------------------------------------------------------------------------
# 1. tool_retrieve_internal
# ---------------------------------------------------------------------------

def tool_retrieve_internal(
    query: str,
    settings: Settings,
    k: Optional[int] = None,
    trace_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k documents from the local ChromaDB vector store.

    Args:
        query:      Free-text biomedical query (e.g. disease name).
        settings:   Loaded Settings object.
        k:          Number of docs to retrieve (defaults to settings.k).
        trace_path: Optional JSONL trace file path.

    Returns:
        List of doc dicts: {id, distance, metadata, text}.
    """
    k = k or settings.k
    docs = retrieve_internal(query, k=k)

    if trace_path:
        doc_details = []
        for d in docs:
            meta = d.get("metadata") or {}
            doc_details.append({
                "id":    d.get("id", "unknown"),
                "year":  meta.get("year", None),
                "title": meta.get("title", d.get("text", "")[:120]),
            })
            
        log_step(
            trace_path,
            step="retrieve",
            params={"query": query, "k": k},
            result={
                "n_docs": len(docs),
                "documents": doc_details,
            },
        )

    return docs


# ---------------------------------------------------------------------------
# 2. tool_evaluate_retrieval
# ---------------------------------------------------------------------------

def tool_evaluate_retrieval(
    query: str,
    docs: List[Dict[str, Any]],
    settings: Settings,
    trace_path: Optional[str] = None,
) -> float:
    """
    Compute a confidence score in [0, 1] for the retrieved doc set.

    Args:
        query:      The original biomedical query.
        docs:       Retrieved docs (output of tool_retrieve_internal).
        settings:   Loaded Settings object.
        trace_path: Optional JSONL trace file path.

    Returns:
        Confidence float in [0, 1].
    """
    confidence = evaluate_retrieval(query, docs)

    if trace_path:
        log_step(
            trace_path,
            step="evaluate",
            params={"query": query, "tau": settings.tau},
            result={
                "confidence": round(confidence, 4),
                "fallback_needed": confidence < settings.tau,
            },
        )

    return confidence


# ---------------------------------------------------------------------------
# 3. tool_fetch_live
# ---------------------------------------------------------------------------

def tool_fetch_live(
    query: str,
    settings: Settings,
    n: Optional[int] = None,
    trace_path: Optional[str] = None,
) -> int:
    """
    Fetch fresh articles from PubMed, embed them, and upsert into ChromaDB.

    Steps:
        1. Call query_pubmed(query, max_results=n).
        2. Build embeddable text: title + abstract.
        3. Embed + upsert into Chroma with PMID in metadata.

    Args:
        query:      Biomedical query for PubMed.
        settings:   Loaded Settings object.
        n:          Number of PubMed records to fetch (defaults to settings.live_fetch_n).
        trace_path: Optional JSONL trace file path.

    Returns:
        Number of new documents successfully upserted.
    """
    try:
        from ls_action_space.action_space import query_pubmed
    except ImportError:
        if _ROOT not in sys.path:
            sys.path.insert(0, _ROOT)
        from ls_action_space.action_space import query_pubmed

    from vectorstore import get_vs

    n = n or settings.live_fetch_n

    if trace_path:
        log_step(
            trace_path,
            step="live_fetch",
            params={"query": query, "n_requested": n},
            result={"status": "started"},
        )

    # 1. Fetch from PubMed
    try:
        articles = query_pubmed(query, max_results=n)
    except Exception as e:
        if trace_path:
            log_step(trace_path, step="live_fetch", params={}, result={"error": str(e)})
        return 0

    # 2. Get ChromaDB collection
    col = get_vs(settings.chroma_path, settings.collection)

    # 3. Embed & upsert
    upserted = 0
    for art in articles:
        pmid = str(art.get("pmid", "")).strip()
        if not pmid:
            continue

        title    = art.get("title", "") or ""
        abstract = art.get("abstract", "") or ""
        text     = f"{title}\n\n{abstract}".strip()
        if not text:
            continue

        doc_id = f"PMID:{pmid}"

        try:
            embedding = _embed_text(text, settings)
            metadata = {
                "pmid":   pmid,
                "title":  title[:500],
                "year":   art.get("year") or 0,
                "source": "pubmed_live",
            }
            col.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
            )
            upserted += 1
        except Exception:
            continue

    if trace_path:
        log_step(
            trace_path,
            step="live_fetch",
            params={"query": query, "n_requested": n},
            result={"n_fetched": len(articles), "n_upserted": upserted},
        )

    return upserted


# ---------------------------------------------------------------------------
# 4. tool_rerank
# ---------------------------------------------------------------------------

def tool_rerank(
    query: str,
    docs: List[Dict[str, Any]],
    settings: Settings,
    k: Optional[int] = None,
    trace_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank retrieved docs by relevance to the query using the LLM.

    The LLM receives a compact list of (index, title, snippet) and returns
    a JSON array of indices in descending relevance order.
    Falls back to the original order if the LLM call fails.

    Args:
        query:      The original biomedical query.
        docs:       Retrieved docs to rerank.
        settings:   Loaded Settings object.
        k:          Number of docs to keep after reranking (defaults to settings.k).
        trace_path: Optional JSONL trace file path.

    Returns:
        Reranked (and possibly truncated) list of docs.
    """
    k = k or settings.k

    if not docs:
        return docs

    # Build a compact representation for the LLM
    entries = []
    for i, d in enumerate(docs):
        meta    = d.get("metadata") or {}
        title   = meta.get("title") or d.get("text", "")[:120]
        snippet = d.get("text", "")[:300].replace("\n", " ")
        entries.append(f'{i}: title="{title}" | snippet="{snippet}"')

    system_prompt = """You are an expert in oncology specialising in drug repurposing.
Your task is to rank a list of drug candidates by relevance."""

    user_prompt = f"""This is the query that you have received : \"{query}\"

Below are {len(docs)} retrieved abstracts, each prefixed by its index.
Return ONLY a JSON array of indices sorted from MOST to LEAST relevant to the query (drug repurposing for the disease).\n"
Example output: [2, 0, 5, 1, 3, 4]\n\n""".join(entries)

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.model,
            max_tokens=300,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
                ],
        )
        raw = resp.choices[0].message.content.strip()
        match = re.search(r"\[[\d,\s]+\]", raw)
        if match:
            order = json.loads(match.group())
            valid_order = [i for i in order if 0 <= i < len(docs)]
            seen = set(valid_order)
            for i in range(len(docs)):
                if i not in seen:
                    valid_order.append(i)
            reranked = [docs[i] for i in valid_order]
        else:
            reranked = docs

    except Exception:
        reranked = docs

    result = reranked[:k]
    
    if trace_path:
        result_details = []
        for r in result:
            meta = r.get("metadata") or {}
            result_details.append({
                "id":    r.get("id", "unknown"),
                "year":  meta.get("year", None),
                "title": meta.get("title", r.get("text", "")[:120]),
            })

        log_step(
            trace_path,
            step="rerank",
            params={"query": query, "n_input": len(docs), "k": k},
            result={
                "n_output": len(result),
                "result":   result_details,
            },
        )

    # Compact top-k retrieval table
    COL_RANK  = 4
    COL_PMID  = 14
    COL_YEAR  = 6
    COL_TITLE = 62
    SEP = f"  {'─'*COL_RANK}  {'─'*COL_PMID}  {'─'*COL_YEAR}  {'─'*COL_TITLE}"

    print(f"\n  {'#':<{COL_RANK}}  {'PMID':<{COL_PMID}}  {'Year':<{COL_YEAR}}  {'Title':<{COL_TITLE}}")
    print(SEP)
    for rank, r in enumerate(result, start=1):
        meta  = r.get("metadata") or {}
        pmid  = str(meta.get("pmid", r.get("id", "—")))[:COL_PMID]
        year  = str(meta.get("year", "—"))[:COL_YEAR]
        title = (meta.get("title") or r.get("text", ""))[:COL_TITLE].replace("\n", " ")
        print(f"  {rank:<{COL_RANK}}  {pmid:<{COL_PMID}}  {year:<{COL_YEAR}}  {title:<{COL_TITLE}}")
    print()
    
    return result


# ---------------------------------------------------------------------------
# 5. tool_extract_candidates
# ---------------------------------------------------------------------------

def tool_extract_candidates(
    disease: str,
    docs: List[Dict[str, Any]],
    settings: Settings,
    trace_path: Optional[str] = None,
) -> List[CandidateDrug]:
    """
    Use the LLM to extract drug repurposing candidates from retrieved docs.

    Steps:
        1. Build a prompt with the top-k doc texts + PMIDs.
        2. Call the LLM; parse JSON response.
        3. Validate each candidate:
           - Must be a single substance or fixed-dose combo.
           - All PMIDs must exist in the retrieval cache.
        4. Return validated List[CandidateDrug].

    Args:
        disease:    Target disease name.
        docs:       Reranked retrieved docs (each with metadata.pmid and text).
        settings:   Loaded Settings object.
        trace_path: Optional JSONL trace file path.

    Returns:
        List of validated CandidateDrug instances (score initialised to 0.0).
    """
    if not docs:
        return []

    pmid_cache = _build_pmid_cache(docs)

    # Build user message: list of abstracts with their PMIDs
    abstract_blocks = []
    for d in docs:
        meta = d.get("metadata") or {}
        pmid = meta.get("pmid", "unknown")
        text = d.get("text", "")[:800]
        abstract_blocks.append(f"[PMID:{pmid}]\n{text}")
    
    system_prompt = """You are a biomedical information extraction expert specialising in drug repurposing.
Your task: extract repurposable drug candidates from the provided abstracts for a target disease.

Rules:
- Only extract SINGLE substances or FIXED-DOSE combinations (e.g. "lopinavir/ritonavir").
- REJECT broad drug classes (e.g. "statins", "NSAIDs"), devices, platforms, biologics classes.
- ONLY use PMIDs that appear in the provided abstracts - never invent PMIDs.
- evidence_tier must be one of: preclinical, phase-1, phase-2, phase-3.
- model must be one of: human, animal, in_vitro.
- outcome must be one of: benefit, mixed, no_effect, harm.
- confidence is YOUR confidence (0.0-1.0) that this candidate is correctly extracted.

Return ONLY a valid JSON array (no markdown, no preamble). Each element:
{
  "drug_name": "...",
  "pmids": ["..."],
  "evidence_tier": "...",
  "model": "...",
  "outcome": "...",
  "mechanism_terms": ["...", "..."],
  "mechanism_hypothesis": "One sentence linking drug mechanism to disease.",
  "rationale": "1-2 sentences grounded in the abstracts.",
  "confidence": 0.0
}
"""


    user_prompt = f"""Target disease: {disease}

Abstracts ({len(docs)} documents):

""".join(abstract_blocks)

    # Call LLM
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=settings.model,
            max_tokens=2000,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            parsed = []

    except Exception as e:
        if trace_path:
            log_step(trace_path, step="extract", params={}, result={"error": str(e)})
        return []

    # Validate and filter candidates
    candidates: List[CandidateDrug] = []
    rejected_drug_class = 0
    rejected_bad_pmid   = 0

    for item in parsed:
        if not isinstance(item, dict):
            continue

        drug_name = str(item.get("drug_name", "")).strip()

        # Filter broad drug classes / devices
        if not drug_name or _is_broad_class_or_non_drug(drug_name):
            rejected_drug_class += 1
            continue

        # Filter PMIDs not in retrieval cache
        raw_pmids   = [str(p).strip() for p in item.get("pmids", []) if str(p).strip()]
        valid_pmids = [p for p in raw_pmids if p in pmid_cache]
        if not valid_pmids:
            rejected_bad_pmid += 1
            continue

        # Build and validate via Pydantic
        try:
            candidate = CandidateDrug(
                drug_name            = drug_name,
                pmids                = valid_pmids,
                evidence_tier        = item.get("evidence_tier", "preclinical"),
                model                = item.get("model", "in_vitro"),
                outcome              = item.get("outcome", "mixed"),
                mechanism_terms      = item.get("mechanism_terms", []),
                mechanism_hypothesis = str(item.get("mechanism_hypothesis", "")),
                rationale            = str(item.get("rationale", "")),
                confidence           = float(item.get("confidence", 0.5)),
                score                = 0.0,
            )
            candidates.append(candidate)
        except Exception:
            continue

    if trace_path:
        log_step(
            trace_path,
            step="extract",
            params={"disease": disease, "n_docs": len(docs)},
            result={
                "n_extracted":         len(candidates),
                "rejected_drug_class": rejected_drug_class,
                "rejected_bad_pmid":   rejected_bad_pmid,
                "pmid_cache_size":     len(pmid_cache),
                "candidates":          [c.model_dump() for c in candidates],
            },
        )

    return candidates