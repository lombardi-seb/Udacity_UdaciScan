"""
scoring.py — Deterministic scoring for UdaciScan repurposing candidates.

Scoring formula (weights from config.yaml):
    score = w.tier       * tier_score
          + w.quality    * quality_score      # model_weight × outcome_weight, clamped to [0,1]
          + w.mechanism  * mechanism_score    # coverage of mechanism_terms (count-based)
          + w.volume     * volume_score       # unique PMIDs, normalised
          + w.confidence * confidence_score   # LLM extractor confidence
          + bonus        if human + benefit + phase-2/3

All intermediate values are clamped to [0, 1] before blending.
The final score is also clamped to [0, 1].

Note: settings.scoring and settings.retrieval are plain dicts (not dataclass instances)
because load_settings() passes the YAML sub-sections directly to Settings().
All accesses therefore use dict-style: settings.scoring["key"].
"""

from __future__ import annotations
from typing import List
from schema import CandidateDrug
from config import Settings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _tier_score(candidate: CandidateDrug, settings: Settings) -> float:
    """Map evidence_tier → weight defined in config.yaml evidence_weights."""
    return _clamp(settings.scoring["evidence_weights"].get(candidate.evidence_tier, 0.0))


def _quality_score(candidate: CandidateDrug, settings: Settings) -> float:
    """
    Study quality proxy: model_weight × outcome_weight.
    Both weights are in [0, 1] per config.yaml, but their product is clamped
    to [0, 1] for safety.
    """
    mw = settings.scoring["model_weights"].get(candidate.model, 0.0)
    ow = settings.scoring["outcome_weights"].get(candidate.outcome, 0.0)
    # outcome_weight can be negative (harm → -0.5); clamp after multiplication
    return _clamp(mw * ow)


def _mechanism_score(candidate: CandidateDrug, settings: Settings) -> float:
    """
    Coverage of mechanism_terms: score rises with the number of distinct terms.
    Normalised by a soft ceiling of 5 terms (enough to reach 1.0).
    Rationale: the LLM extractor already grounded these terms in retrieved text,
    so more terms = stronger mechanistic support.
    """
    n_terms = len(set(t.strip().lower() for t in candidate.mechanism_terms if t.strip()))
    SOFT_CEILING = 5
    return _clamp(n_terms / SOFT_CEILING)


def _volume_score(candidate: CandidateDrug, settings: Settings) -> float:
    """
    Evidence volume: number of unique supporting PMIDs, normalised by
    live_fetch_n (the max number of records pulled in one live fetch).
    Represents how well-supported the candidate is across distinct papers.
    """
    n_unique = len(set(candidate.pmids))
    normaliser = max(1, settings.live_fetch_n)
    return _clamp(n_unique / normaliser)


def _bonus(candidate: CandidateDrug, settings: Settings) -> float:
    """
    Extra credit when there is human benefit evidence at phase-2 or phase-3.
    Value comes from config.yaml bonus.human_benefit_phase2plus.
    """
    if (
        candidate.model == "human"
        and candidate.outcome == "benefit"
        and candidate.evidence_tier in ("phase-2", "phase-3")
    ):
        return settings.scoring["bonus"].get("human_benefit_phase2plus", 0.0)
    return 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidate(candidate: CandidateDrug, settings: Settings) -> float:
    """
    Compute a deterministic score in [0, 1] for a single CandidateDrug.

    Args:
        candidate:  A CandidateDrug instance (fields already validated by Pydantic).
        settings:   Loaded Settings object (from config.py / config.yaml).

    Returns:
        A float score in [0, 1].
    """
    w = settings.scoring["weights"]

    tier        = _tier_score(candidate, settings)
    quality     = _quality_score(candidate, settings)
    mechanism   = _mechanism_score(candidate, settings)
    volume      = _volume_score(candidate, settings)
    confidence  = _clamp(candidate.confidence)

    raw_score = (
        w.get("tier",       0.0) * tier
      + w.get("quality",    0.0) * quality
      + w.get("mechanism",  0.0) * mechanism
      + w.get("volume",     0.0) * volume
      + w.get("confidence", 0.0) * confidence
      + _bonus(candidate, settings)
    )

    return _clamp(raw_score)


def score_and_rank(
    candidates: List[CandidateDrug],
    settings: Settings,
) -> List[CandidateDrug]:
    """
    Score every candidate in place, then return them sorted by score (desc).
    Ties are broken deterministically by drug_name (alphabetical asc).

    Args:
        candidates: List of CandidateDrug instances.
        settings:   Loaded Settings object.

    Returns:
        The same list, with .score updated and sorted descending.
    """
    for c in candidates:
        c.score = score_candidate(c, settings)

    return sorted(candidates, key=lambda c: (-c.score, c.drug_name))
