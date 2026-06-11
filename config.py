# Loads and defines configuration settings and scoring weights from a YAML file.
import os, yaml
from dataclasses import dataclass
from typing import Dict

@dataclass
class RetrievalCfg:
    """Parameters for confidence estimation.

    top_n_for_mean:             how many top results to consider for the mean
    recency_window_years:   a paper ≤ this many years old contributes to the recency boost
    weights:                max/mean/margin/recency
    """
    top_n_for_mean: int
    recency_window_years: int
    weights: Dict[str, float]  # keys: max/mean/margin/recency

@dataclass
class ScoringCfg:
    """Tunable scoring parameters for ranking repurposing candidates.

    evidence_weights: maps trial tier → relative strength (0..1).
    model_weights:    maps evidence model → preference (human > animal > in_vitro).
    outcome_weights:  maps effect direction → utility (benefit > mixed > no_effect; harm < 0).
    weights:          blend coefficients for tier/quality/mechanism/volume/confidence.
    bonus:            extra credit for particularly persuasive evidence patterns.
    """
    evidence_weights: Dict[str, float]
    model_weights: Dict[str, float]
    outcome_weights: Dict[str, float]
    weights: Dict[str, float]
    bonus: Dict[str, float]

@dataclass
class Settings:
    """Top-level agent settings.

    chroma_path:        path to store ChromaDB files
    collection:         name of ChromaDB collection
    model:              name of OpenAI LLM to use
    embed_model:        name of OpenAI embedding model to use
    email:              email address (needed as identifier for NCBI Entrez queries)
    k:                  top-k retrieval from Chroma after reranking
    tau:                confidence threshold to trigger live PubMed fetch
    max_attempts:       live-retrieval retries
    live_fetch_n:       how many PubMed records to pull when fetching live
    scoring:            Tunable scoring parameters for ranking repurposing candidates.
    """
    chroma_path: str
    collection: str
    model: str
    embed_model: str
    email: str
    k: int
    tau: float
    max_attempts: int
    live_fetch_n: int
    retrieval: RetrievalCfg
    scoring: ScoringCfg

def load_settings(path="config.yaml") -> Settings:
    with open(path) as f:
        y = yaml.safe_load(f)
    settings = Settings(**y["pharmascan"])
    os.environ["ENTREZ_EMAIL"] = settings.email
    return settings