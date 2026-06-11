# UdaciScan — Project Report

---
## 1. Deviations from the starter kit

Two files absent from the starter kit - `llm_utils.py` and `vectorstore.py` - were sourced from a public GitHub repository and added to the project. A related question was posted on the Udacity Knowledge forum but received no response (https://knowledge.udacity.com/questions/1084317).

Additionally, config.yaml contained characters that caused parsing errors in the local environment (e.g. `≤` replaced with `<=`); these were corrected before use.

---

## 2. What the Solution Does
 
UdaciScan is a stateful AI research agent that answers drug repurposing queries for a target disease. Given a disease name, the agent:
 
1. **Retrieves** the most relevant biomedical abstracts from a local ChromaDB vector store seeded with PubMed records.
2. **Evaluates** retrieval confidence using a weighted blend of max similarity, mean similarity, margin, and document recency.
3. **Falls back** to live PubMed retrieval via `query_pubmed()` when confidence falls below a threshold τ, then upserts the new abstracts into ChromaDB and re-queries.
4. **Reranks** the top-k documents using the LLM to prioritise the most relevant abstracts for the specific disease.
5. **Extracts** drug repurposing candidates using structured LLM information extraction, producing normalised drug names, evidence tiers, study context, mechanism terms, rationales, and supporting PMIDs.
6. **Scores** candidates deterministically by blending five features (evidence tier, study quality, mechanism coverage, evidence volume, extractor confidence) with weights from `config.yaml`, plus a bonus for human benefit at Phase 2+.
7. **Outputs** a `RepurposingBrief` validated by Pydantic, serialised to JSON and rendered to Markdown with clickable PubMed citation links, and logs a full trajectory trace in JSONL format.

---
 
## 3. Confidence Threshold Justification (τ)
 
### How τ was obtained
 
τ was estimated using the `estimate_tau()` function in `retrieval_tools.py`. The function:
 
1. Runs `retrieve_internal()` on a labelled set of **12 in-domain queries** (diseases well-represented in the vector store, e.g. Parkinson's disease, Glioblastoma, NASH) and **9 out-of-domain queries** (rare diseases with low or no coverage, e.g. Ehlers-Danlos syndrome, POEMS syndrome, Gaucher Disease).
2. Computes a confidence score via `evaluate_retrieval()` for each query — a weighted blend of max similarity, mean similarity over top-5, similarity margin, and a recency indicator.
3. Fits a logistic regression classifier (no regularisation) on the resulting scores with binary labels (1 = in-domain, 0 = out-of-domain).
4. Solves for the **decision boundary** where P(in-domain | score) = 0.5, i.e. `τ = −intercept / coefficient`.

The resulting value was **τ = 0.660**, which is set in `config.yaml`. The observed score distributions were:
 
| Group | Mean confidence | Min / Max |
|---|---|---|
| In-domain (12 queries) | 0.687 | min = 0.664 |
| Out-of-domain (9 queries) | 0.626 | max = 0.654 |
 
### Why this is a reasonable operating point
 
A threshold of 0.660 sits precisely between the two distributions: the minimum in-domain confidence (0.664) is higher than the maximum out-of-domain confidence (0.654), meaning the two groups are perfectly linearly separable at this threshold. This makes τ = 0.660 an optimal operating point with zero misclassification on the calibration set — in-domain queries always score above τ and are answered from internal knowledge, while out-of-domain queries always score below τ and correctly trigger the live PubMed fallback. The logistic regression decision boundary is mathematically grounded rather than manually tuned, making it reproducible and justifiable.
 
### Trade-offs
 
A higher τ (e.g. 0.75) would increase fallback frequency, fetching more live literature and improving coverage, but at the cost of higher API usage, longer runtime, and potential noise from less-relevant PubMed results. A lower τ (e.g. 0.55) would reduce fallback frequency, making the agent faster and cheaper, but risking low-quality answers for diseases at the edge of the vector store's coverage. The chosen value of 0.660 balances precision (avoiding unnecessary fallbacks for in-domain queries) with recall (ensuring fallback is triggered when internal evidence is genuinely weak).

---
 
## 4. Handling Hallucinations
 
UdaciScan addresses hallucination risk at three levels:
 
### 4.1 Citation validation (PMID grounding)
 
After the LLM extracts candidates, every cited PMID is checked against the **retrieval cache** — the set of PMIDs actually present in the documents provided to the LLM. Any PMID not found in this cache is silently removed from the candidate's citation list. If a candidate ends up with zero valid PMIDs after this filter, it is **rejected entirely** and does not appear in the output. This is logged in the trace file (`rejected_bad_pmid` counter) as auditable proof that the filter ran.
 
### 4.2 Drug class and entity filtering
 
The LLM occasionally extracts broad drug classes ("statins", "NSAIDs"), non-pharmacological interventions, or platform technologies instead of specific drugs. 
To block these systematically, a three-layer filter is applied to every extracted drug_name before Pydantic validation:
1. Exact class list : a curated set of known class names and abbreviations (e.g. "statins") is matched case-insensitively against the normalised name.
2. Suffix pattern : a regex detects names ending in generic suffixes that signal a class rather than a substance (e.g. "inhibitors", "antagonists").
3. Non-pharmacological keywords : a second regex blocks names containing words associated with devices, non-pharmacological interventions (e.g. "device", "surgery"). 

Names of 4 or more words without "/" are also rejected as likely descriptions rather than drug names. Fixed-dose combos (e.g., "lopinavir/ritonavir") are allowed. 
Rejected entries are counted in the trace (`rejected_drug_class`).
 
### 4.3 Pydantic schema enforcement
 
All extracted candidates must pass Pydantic validation against `CandidateDrug`. Fields like `evidence_tier`, `model`, and `outcome` are typed as `Literal` enumerations — any value not in the allowed set causes a `ValidationError` and the candidate is dropped. Score and confidence fields are bounded to `[0.0, 1.0]`. This makes the schema the last line of defence against malformed LLM output reaching the final brief.

---

## 5. Agent Run Results

The agent was executed on four diseases in a single command (see `screenshots/` for the full terminal output):

```
python udaciscan/agent.py --diseases "Alzheimer's disease" "Multiple sclerosis" "Nonalcoholic steatohepatitis" "Sickle Cell Disease"
``` 

All runs used `k=12` and `τ=0.66`. The table below summarises the outcome for each disease.

| Disease                      | Confidence | Fallback triggered | Docs upserted | Post-upsert confidence | Candidates extracted |
|------------------------------|------------| --- | --- |------------------------|----------------------|
| Alzheimer's disease          | 0.7156     | No | — | —                      | 2                    |
| Multiple sclerosis           | 0.6776     | No | — | —                      | 3                    |
| Nonalcoholic steatohepatitis | 0.6645     | No | — | —                      | 3                    |
| Stickle cell disease         | 0.6215     | Yes | 25 | 0.6914                 | 3                    |

The Stickle cell disease run demonstrates the fallback mechanism working as intended: initial confidence fell below τ, triggering a live PubMed fetch that upserted 25 new abstracts into ChromaDB. Re-querying after upsert raised confidence from 0.6215 to 0.6914, above the threshold, before proceeding to reranking and candidate extraction.

---

