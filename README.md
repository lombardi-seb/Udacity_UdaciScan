# UdaciScan 🔬
### An AI Research Agent for Drug-Repurposing Insights

> Capstone project for **Course 3 – Building Agents with Core Bioinformatics Tools**  
> [Udacity Nanodegree: Agentic AI for Life Sciences (ND903)](https://www.udacity.com/course/agentic-ai-for-life-sciences--nd903)

---

## Overview

**UdaciScan** is a Python-based agentic RAG system that acts as an AI "research scout" for drug repurposing. Given a disease as input, the agent autonomously searches a local biomedical knowledge base (ChromaDB), falls back to live PubMed literature when confidence is low, and produces a **structured, source-cited drug-repurposing brief** — deterministically scored, Pydantic-validated, and fully traceable.

Typical queries the agent is designed to answer:

- *"What repurposing candidates look promising for **Parkinson's disease** right now?"*
- *"Is there clinical (phase-1/2/3) evidence for **metformin** in **NASH/MASLD**?"*
- *"Summarize top candidates for **glioblastoma** and cite the strongest papers."*

---

## Key Features

- 🔁 **Two-tier retrieval** — primary RAG over a local ChromaDB seeded with PubMed abstracts; automatic fallback to live PubMed queries when the local store is insufficient
- 📊 **Confidence gating (τ)** — a scalar confidence score derived from retrieval signals (similarity, recency, margin) controls whether the fallback is triggered
- 🔄 **Live fetch + upsert** — new PubMed records are embedded and persisted into ChromaDB, improving all subsequent queries on the same disease
- 📐 **Deterministic scoring** — candidates are ranked by a configurable blend of evidence tier, study model/outcome, mechanism coverage, evidence volume, and extractor confidence
- 📋 **Pydantic-validated output** — every brief is validated through a `RepurposingBrief` schema before being serialized to JSON and Markdown
- 🔒 **Anti-hallucination enforcement** — all PMIDs in the output must originate from the retrieved document cache; no external citations allowed
- 📝 **Full audit trail** — every agent run produces a JSONL trajectory log (tool calls, parameters, timings, decisions)
- ⚙️ **Parameterized configuration** — models, thresholds (τ), and scoring weights are managed in `config.yaml`

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | [LangChain](https://www.langchain.com/) |
| Vector database | [ChromaDB](https://www.trychroma.com/) |
| LLM backbone | OpenAI API |
| Structured outputs | [Pydantic](https://docs.pydantic.dev/) |
| Live literature | `query_pubmed()` (PubMed API) |
| PDF ingestion | `extract_pdf_content()` (optional) |
| Configuration | YAML (`config.yaml`) |
| Language | Python 3.10+ |

---

## Project Structure

```
Udacity_UdaciScan/
│
├── udaciscan/                      # Main agent package
│   ├── agent.py                    # Stateful 7-step agent pipeline
│   ├── schema.py                   # Pydantic models (RepurposingBrief, Candidate, …)
│   ├── scoring.py                  # Deterministic candidate scoring & ranking
│   └── reporting.py                # Markdown & JSON brief rendering
│
├── ls_action_space/
│   └── action_space.py             # query_pubmed() + extract_pdf_content() helpers
│
├── retrieval_tools.py              # retrieve_internal(), evaluate_retrieval(), estimate_tau()
│
├── data/
│   └── chroma/                     # Pre-loaded vector store (≥1,000 PubMed records)
│
├── outputs/                        # Generated briefs (git-ignored or committed per run)
│   ├── brief_Parkinsons.json
│   ├── brief_Parkinsons.md
│   ├── brief_NASH_MASLD.json
│   ├── brief_NASH_MASLD.md
│   ├── brief_Glioblastoma.json
│   └── brief_Glioblastoma.md
│
├── runs/                           # Trajectory logs
│   └── trace_<timestamp>.jsonl
│
├── config.yaml                     # Models, k, τ, scoring weights
├── requirements.txt
├── REPORT.md                       # Threshold justification & design notes
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/lombardi-seb/Udacity_UdaciScan.git
cd Udacity_UdaciScan

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Configuration

Edit `config.yaml` to tune the agent behavior:

```yaml
model: gpt-4o-mini
embedding_model: text-embedding-3-small
k: 10                  # Number of docs retrieved internally
tau: 0.72              # Confidence threshold — below this, live fetch is triggered
n_live: 5              # Number of PubMed records to fetch on fallback

scoring:
  weight_evidence_tier: 0.30
  weight_model_outcome: 0.25
  weight_mechanism: 0.20
  weight_volume: 0.15
  weight_extractor_confidence: 0.10
  bonus_human_phase2: 0.05
```

### Running the Agent

```bash
# Run on a single disease
python -m udaciscan.agent --disease "Parkinson's disease"

# Run on the full demo set (≥3 diseases)
python -m udaciscan.agent --disease "Parkinson's disease" "NASH/MASLD" "Glioblastoma"
```

Output files are written to `outputs/` and `runs/` automatically.

---

## Agent Pipeline

The agent follows a stateful 7-step workflow:

```
User Query (disease name)
        │
        ▼
 ┌─────────────────────────┐
 │  Step 1 · Retrieve      │  ChromaDB similarity search (top-k)
 └──────────┬──────────────┘
            │
            ▼
 ┌─────────────────────────┐
 │  Step 2 · Evaluate      │  Compute confidence scalar from
 │  Confidence             │  similarity scores, recency, margin
 └──────────┬──────────────┘
            │
     confidence < τ ?
      ├─ YES ──────────────►  ┌──────────────────────────────┐
      │                       │  Step 3 · Live Fetch & Upsert│
      │                       │  query_pubmed() → embed →    │
      │                       │  upsert into ChromaDB →      │
      │                       │  re-query → recompute conf.  │
      │                       └──────────────┬───────────────┘
      └─ NO ◄──────────────────────────────────┘
            │
            ▼
 ┌─────────────────────────┐
 │  Step 4 · Rerank        │  LLM-assisted or heuristic reranking
 └──────────┬──────────────┘
            │
            ▼
 ┌─────────────────────────┐
 │  Step 5 · Extract       │  LLM information extraction:
 │  Candidates             │  drug name, PMIDs, evidence tier,
 │                         │  mechanism terms, study context
 └──────────┬──────────────┘
            │  Validate: single substances only,
            │  all PMIDs from retrieved cache
            ▼
 ┌─────────────────────────┐
 │  Step 6 · Score & Rank  │  Deterministic blended score (scoring.py)
 └──────────┬──────────────┘
            │
            ▼
 ┌─────────────────────────────────────────┐
 │  Step 7 · Synthesize Brief              │
 │  RepurposingBrief (Pydantic) →          │
 │  outputs/brief_<Disease>.json           │
 │  outputs/brief_<Disease>.md             │
 │  runs/trace_<timestamp>.jsonl           │
 └─────────────────────────────────────────┘
```

---

## Output Format

### JSON (`outputs/brief_<Disease>.json`)

```json
{
  "disease": "Parkinson's disease",
  "date": "2025-06-01",
  "method_summary": "Internal ChromaDB retrieval (k=10, τ=0.72); live PubMed fallback triggered; 5 records upserted.",
  "candidates": [
    {
      "drug_name": "Nilotinib",
      "rank": 1,
      "score": 0.87,
      "evidence_tier": "phase-2",
      "mechanism_hypothesis": "BCR-ABL inhibitor shown to activate autophagy and reduce α-synuclein aggregation.",
      "rationale": "Two phase-2 trials in PD patients reported safety and biomarker improvement.",
      "model": "human",
      "outcome": "benefit",
      "pmids": ["34201234", "35891045"]
    }
  ],
  "limitations": [
    "Coverage limited to PubMed abstracts — full-text evidence not systematically included.",
    "This output is not medical advice."
  ]
}
```

### Markdown (`outputs/brief_<Disease>.md`)

Human-readable report with ranked candidates, rationales, evidence tiers, and clickable PMID citations.

### Trajectory log (`runs/trace_<timestamp>.jsonl`)

One JSON object per line, recording each agent step — tool called, parameters, document counts, timing, and decisions (e.g., whether fallback was triggered and why).

---

## Confidence Threshold (τ) Justification

The threshold τ was selected by running `estimate_tau()` from `retrieval_tools.py` on a mixed set of in-domain (drug-repurposing) and out-of-domain queries, then choosing the operating point that balances:

- **Precision** — avoiding unnecessary PubMed calls when the local store is already sufficient
- **Recall** — ensuring live fetch is triggered whenever the retrieval confidence is genuinely low

A full justification (method, chosen value, trade-offs) is provided in [`REPORT.md`](./REPORT.md).

---

## Quality Constraints

| Rule | Details |
|---|---|
| No hallucinated citations | All PMIDs must appear in the retrieved document cache |
| Grounded rationales | 1–2 sentences per candidate, directly supported by retrieved text |
| Validated schema | Output rejected by Pydantic if any required field is missing or malformed |
| Single substances only | Drug classes, devices, and platforms are filtered out during candidate extraction |
| Non-medical-advice disclaimer | Included in every brief's `limitations` section |

---

## Demo Diseases

The agent is validated on at least three diseases:

| Disease | Fallback triggered? | Records upserted |
|---|---|---|
| Parkinson's disease | — | — |
| NASH / MASLD | — | — |
| Glioblastoma | — | — |

*(Fill in with actual run results)*

---

## Learning Context

This project is the capstone of **Course 3** of the Udacity *Agentic AI for Life Sciences* Nanodegree, which covers:

- Building agents with LangChain tools and external APIs
- Structured outputs with Pydantic
- Agent state management with LangGraph
- Short- and long-term agent memory
- Agentic RAG with ChromaDB
- Web search and live literature agents
- Agent evaluation (citation integrity, retrieval quality, entity accuracy)

---

## License

This project was developed as part of a Udacity Nanodegree program. Please refer to Udacity's academic integrity policies before reusing any part of this code in other educational submissions.

---

## Acknowledgements

- [Udacity – Agentic AI for Life Sciences](https://www.udacity.com/course/agentic-ai-for-life-sciences--nd903)
- [LangChain](https://www.langchain.com/) & [ChromaDB](https://www.trychroma.com/) teams
- [PubMed / NCBI](https://pubmed.ncbi.nlm.nih.gov/) for the biomedical literature API
