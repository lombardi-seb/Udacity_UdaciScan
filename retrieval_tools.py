# Provides tools for vector retrieval and for evaluating the quality of retrieved documents.
from __future__ import annotations
from typing import List, Dict, Any
from datetime import date
from vectorstore import get_vs
from config import load_settings
import os, re
from tqdm import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
import numpy as np

from llm_utils import embed

settings = load_settings()

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _to_similarity(distance: float | None, similarity: float | None = None) -> float:
    # Prefer native similarity; otherwise map cosine distance [0,2] -> sim [0,1]
    if similarity is not None:
        return _clamp(float(similarity))
    if distance is not None:
        return _clamp(1.0 - (float(distance) / 2.0))
    return 0.0

def _doc_year(meta: Dict[str, Any] | None) -> int:
    try:
        return int((meta or {}).get("year", 0)) or 0
    except Exception:
        return 0

def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", s.lower())

def _disease_coverage(disease: str, docs: List[Dict[str, Any]], n: int) -> float:
    """
    Fraction of the top-n docs that clearly talk about the disease.
    Signals: exact disease phrase OR MeSH disease term match.
    """
    if n == 0:
        return 0.0
    dz = _normalize(disease).strip()
    hits = 0
    for d in docs[:n]:
        text = _normalize(d.get("text", "")[:4000])  # cap for speed
        mesh = " ".join((d.get("metadata", {}) or {}).get("mesh", []) or []).lower()
        if (dz and dz in text) or (dz and dz in mesh):
            hits += 1
    return hits / n

def _uniqueness_ratio(docs: List[Dict[str, Any]], n: int) -> float:
    """
    How many *distinct* PMIDs among the top-n? (1.0 = all unique)
    Helps penalize clusters of dupes.
    """
    if n == 0:
        return 0.0
    pmids = []
    for d in docs[:n]:
        pmid = (d.get("metadata", {}) or {}).get("pmid")
        if not pmid and isinstance(d.get("id"), str) and d["id"].startswith("PMID:"):
            pmid = d["id"].split("PMID:")[-1]
        if pmid:
            pmids.append(str(pmid))
    return len(set(pmids)) / max(1, n)

def evaluate_retrieval(_query: str, docs: List[Dict[str, Any]]) -> float:
    """
    Confidence in [0,1] = weighted blend of:
      - max similarity among retrieved docs
      - mean similarity of top-N docs
      - recency indicator (≥1 doc within recency_window_years)
    Minimal, easy-to-tune, and cheap to maintain.
    """
    if not docs:
        return 0.0

    top_n_for_mean = settings.retrieval["top_n_for_mean"]
    recency_window_years = settings.retrieval["recency_window_years"]
    w = settings.retrieval["weights"]

    sims = sorted((_to_similarity(distance=d["distance"]) for d in docs), reverse=True)
    top_n = max(1, min(top_n_for_mean, len(sims)))
    max_sim = sims[0]
    mean_sim = sum(sims[:top_n]) / top_n

    # margin: separation between the best hit and the pack
    margin = max(0.0, max_sim - mean_sim)

    # Recency = 1 if any top-N doc is within window; else 0 (soft version: use fraction)
    this_year = date.today().year
    years = [_doc_year(d.get("metadata")) for d in docs[:top_n]]
    recent_hits = [y for y in years if y and (this_year - y) <= recency_window_years]
    recency = 1.0 if recent_hits else 0.0

    conf = (
        w.get("max")      * max_sim +
        w.get("mean")     * mean_sim +
        w.get("margin")   * margin +
        w.get("recency")  * recency
    )
    return max(0.0, min(1.0, conf))

def retrieve_internal(query: str, k: int = settings.k) -> List[Dict[str, Any]]:
    k = k if k and k >= 1 else settings.k
    col = get_vs(settings.chroma_path, settings.collection)
    q_emb = embed([query])
    res = col.query(query_embeddings=q_emb, n_results=k, include=["distances","metadatas","documents"])
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "id": res["ids"][0][i],
            "distance": res["distances"][0][i],     # cosine distance (lower is better)
            "metadata": res["metadatas"][0][i],
            "text": res["documents"][0][i],
        })
    return docs


def estimate_tau(n_matches: int = 5) -> float:
    # Generate diverse queries - some in-domain, some out
    in_domain = [
        "Parkinson's disease",
        "Alzheimer's disease",
        "Amyotrophic lateral sclerosis",
        "Multiple sclerosis",
        "Idiopathic pulmonary fibrosis",
        "Nonalcoholic steatohepatitis",
        "Glioblastoma",
        "Ovarian cancer",
        "Pancreatic cancer",
        "Ulcerative colitis",
        "Systemic lupus erythematosus",
        "Type 2 diabetes mellitus",
    ]
    out_of_domain = ["Lymphoma","Ehlers-Danlos syndrome", "POEMS syndrome", "Erdheim-Chester disease", "Sickle Cell Disease",
                    "Cystic Fibrosis", "Duchenne Muscular Dystrophy", "Gaucher Disease", "Haemophilia, Phenylketonuria"]

    # Get confidence scores for labeled examples
    in_scores = []
    for query in in_domain:
        docs = retrieve_internal(query, k=n_matches)
        in_scores.append(evaluate_retrieval(query, docs))

    out_scores = []
    for query in out_of_domain:
        docs = retrieve_internal(query, k=n_matches)
        out_scores.append(evaluate_retrieval(query, docs))

    X = np.array(in_scores+out_scores).reshape(-1,1)
    y = [1]*len(in_scores) + [0]*len(out_scores)

    # Fit logistic regression
    clf = LogisticRegression(penalty=None)
    clf.fit(X, y)

    # For logistic regression, the decision boundary where P(y=1|X) is 0.5
    # occurs where the linear model (w*x + b) equals 0.
    # We solve for x (the confidence score) to find our threshold, tau.
    tau = -clf.intercept_[0] / clf.coef_[0, 0]

    print(f"Tau = {tau:.3f}")
    print(f"Mean & Min in-domain conf: ",np.mean(in_scores),np.min(in_scores))
    print(f"Mean & Max out-domain confs: ",np.mean(out_scores),np.max(out_scores))

    return float(tau)


if __name__ == "__main__":
    c_tau = estimate_tau()
    print(c_tau)

    for query in ["Parkinson's disease", "Ehlers-Danlos"]: # Ehlers-Danlos is a rare disese not in the database
        docs = retrieve_internal(query)
        print(docs)
        C = evaluate_retrieval(query, docs)
        print(query,C)
    print("done!")