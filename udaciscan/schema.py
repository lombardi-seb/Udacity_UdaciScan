"""
schema.py — Pydantic models for UdaciScan.
 
Defines the structured output of the agent:
  - CandidateDrug  : a single repurposing candidate with evidence and score
  - RepurposingBrief : the full output brief for a given disease
"""
 
from __future__ import annotations
from typing import List, Literal
from datetime import date
from pydantic import BaseModel, Field, field_validator
 
 
class CandidateDrug(BaseModel):
    """A single drug repurposing candidate extracted from retrieved literature."""
 
    drug_name: str = Field(
        ...,
        description="Normalized drug name (single substance or fixed-dose combo only)."
    )
    pmids: List[str] = Field(
        ...,
        min_length=1,
        description="PMIDs supporting this candidate — must all exist in the retrieval cache."
    )
    evidence_tier: Literal["preclinical", "phase-1", "phase-2", "phase-3"] = Field(
        ...,
        description="Highest evidence tier found in the retrieved literature."
    )
    model: Literal["human", "animal", "in_vitro"] = Field(
        ...,
        description="Best study model available (human > animal > in_vitro)."
    )
    outcome: Literal["benefit", "mixed", "no_effect", "harm"] = Field(
        ...,
        description="Direction/quality of the observed effect."
    )
    mechanism_terms: List[str] = Field(
        default_factory=list,
        description="Biological targets or pathways linking the drug to the disease."
    )
    mechanism_hypothesis: str = Field(
        ...,
        description="One-sentence mechanistic hypothesis (pathway/target → disease link)."
    )
    rationale: str = Field(
        ...,
        description="1–2 sentence rationale grounded in retrieved text."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="LLM extractor confidence score in [0, 1]."
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Final deterministic score in [0, 1] (computed by scoring.py)."
    )
 
    @field_validator("pmids")
    @classmethod
    def pmids_must_be_non_empty_strings(cls, v: List[str]) -> List[str]:
        for pmid in v:
            if not pmid or not str(pmid).strip():
                raise ValueError("Each PMID must be a non-empty string.")
        return [str(p).strip() for p in v]
 
    @field_validator("drug_name")
    @classmethod
    def drug_name_must_be_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("drug_name must be a non-empty string.")
        return v.strip()
 
 
class RepurposingBrief(BaseModel):
    """Full repurposing brief for a target disease, produced by the UdaciScan agent."""
 
    disease: str = Field(
        ...,
        description="Target disease name as provided by the user."
    )
    date: str = Field(
        default_factory=lambda: date.today().isoformat(),
        description="Date the brief was generated (ISO 8601 format)."
    )
    method_summary: str = Field(
        ...,
        description="Short description of the retrieval and extraction method used."
    )
    candidates: List[CandidateDrug] = Field(
        default_factory=list,
        description="Ranked list of repurposing candidates (descending score)."
    )
    limitations: str = Field(
        ...,
        description="Known limitations: coverage gaps, PDF availability, disclaimer, etc."
    )
 
    @field_validator("candidates")
    @classmethod
    def candidates_sorted_by_score(cls, v: List[CandidateDrug]) -> List[CandidateDrug]:
        return sorted(v, key=lambda c: c.score, reverse=True)