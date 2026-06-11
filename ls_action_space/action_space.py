"""
Minimal shared life‑sciences action space for agentic LLMs

This module provides a robust, production-grade set of tools for life sciences
research, designed for reliable use by LLM-based agents. It features a
domain-aware rate limiter, structured JSON outputs, and resilient error handling.

It exposes a compact set of tools covering:
- Literature retrieval (PubMed, arXiv, Scholar)
- Evidence extraction (PDF + web)
- Clinical/genetics reasoning (ClinVar, OMIM)
- Sequence analytics (NCBI BLAST)
- Clinical trial interrogation (ClinicalTrials.gov)
- Local pharmacovigilance + knowledge bases (FAERS CSVs, OMIM.csv, ClinVar TSV)
- A persistent REPL for quick data work (with optional Bash/R shebangs)

Nine core callables used by agents:
    query_pubmed, query_scholar, query_arxiv,
    extract_pdf_content, extract_url_content,
    query_clinvar, query_omim,
    blast_sequence,
    run_python_repl
Plus: query_clinicaltrials (live endpoint) and 3 local-KB helpers:
    query_local_clinvar, query_local_omim, query_local_faers

Dependencies (install what you need):
    pip install requests pandas beautifulsoup4 lxml biopython pypdf2 arxiv
Optional (improve extraction):
    pip install readability-lxml feedparser

Environment variables:
    ENTREZ_EMAIL          -> your.email@example.com  (required by NCBI E-utilities)
    NCBI_API_KEY          -> optional E-utilities key (higher rate limits)
    OMIM_API_KEY          -> required for live OMIM API
    ACTIONSPACE_USER_AGENT-> optional, sent to remote services where allowed

Local data (paths can be overridden per-call):
    ./data/clinvar_snapshot.tsv
    ./data/omim_catalog.csv
    ./data/faers/faers_*.csv

Note:
- Google Scholar scraping is inherently brittle; it’s included as a best‑effort backup.
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
import time
import math
import shutil
import logging
from io import BytesIO, StringIO
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

# --- Environment variables  -------------------------------------------------

from dotenv import load_dotenv
load_dotenv(dotenv_path="entrez.env")

# --- Per-domain rate limiter (polite API usage) -----------------------------
from urllib.parse import urlparse

class RateLimiter:
    def __init__(self, intervals: Dict[str, float]):
        self.intervals = intervals
        self.last: Dict[str, float] = {}

    def wait(self, key: str) -> None:
        interval = self.intervals.get(key, self.intervals.get("generic", 0.0))
        if interval <= 0:
            return
        now = time.monotonic()
        prev = self.last.get(key)
        if prev is not None:
            sleep_for = interval - (now - prev)
            if sleep_for > 0:
                time.sleep(sleep_for)
        self.last[key] = time.monotonic()

# NCBI: 3 req/s without key (~0.34s), ~10 req/s with key (~0.1s)
_ncbi_interval = 0.1 if os.getenv("NCBI_API_KEY") else 0.34
_limiter = RateLimiter({
    "ncbi": _ncbi_interval,
    "omim": 0.25,     # ~4 req/s
    "ctgov": 0.20,    # ~5 req/s
    "scholar": 5.0,   # be extra gentle
    "arxiv": 0.30,    # ~3 req/s
    "generic": 0.20,  # safe default for others
})

# Optional libraries (import lazily and guard usage)
try:
    from Bio import Entrez
    from Bio.Blast import NCBIWWW, NCBIXML
except Exception:  # pragma: no cover
    Entrez = None  # type: ignore
    NCBIWWW = None  # type: ignore
    NCBIXML = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    import PyPDF2
except Exception:  # pragma: no cover
    PyPDF2 = None  # type: ignore

try:
    import arxiv as arxiv_py
except Exception:  # pragma: no cover
    arxiv_py = None  # type: ignore

try:  # optional, nicer web extraction
    from readability import Document as ReadabilityDoc  # type: ignore
except Exception:  # pragma: no cover
    ReadabilityDoc = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration & HTTP utilities
# ---------------------------------------------------------------------------
USER_AGENT = os.getenv(
    "ACTIONSPACE_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
)

NCBI_EMAIL = os.getenv("ENTREZ_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

if Entrez is not None:
    if not NCBI_EMAIL:
        logging.warning(
            "ENTREZ_EMAIL not set. Please export ENTREZ_EMAIL to comply with NCBI policy."
        )
    Entrez.email = NCBI_EMAIL or "anonymous@example.com"
    if NCBI_API_KEY:
        Entrez.api_key = NCBI_API_KEY

DEFAULT_TIMEOUT = 30

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(max_retries=3)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)
_default_headers = {"User-Agent": USER_AGENT}


def _http_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    stream: bool = False,
) -> requests.Response:
    """HTTP GET with retries, timeouts, UA, and polite rate-limiting per domain."""
    def _rate_key_for_url(u: str) -> str:
        host = urlparse(u).netloc.lower()
        if host.endswith("ncbi.nlm.nih.gov"):
            return "ncbi"
        if host == "api.omim.org":
            return "omim"
        if host.endswith("clinicaltrials.gov"):
            return "ctgov"
        if host == "scholar.google.com":
            return "scholar"
        if host.endswith("arxiv.org"):
            return "arxiv"
        return "generic"

    _limiter.wait(_rate_key_for_url(url))
    h = dict(_default_headers)
    if headers:
        h.update(headers)
    resp = _session.get(url, params=params, headers=h, timeout=timeout, stream=stream)
    resp.raise_for_status()
    return resp



def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _entrez_call(method: str, /, **kwargs):
    """Rate-limited wrapper around Bio.Entrez calls."""
    if Entrez is None:
        raise ImportError("biopython is required for NCBI E-utilities")
    _limiter.wait("ncbi")
    fn = getattr(Entrez, method)
    return fn(**kwargs)

# ---------------------------------------------------------------------------
# 1) Literature retrieval
# ---------------------------------------------------------------------------

def query_pubmed(
    query: str,
    max_results: int = 20,
    include_mesh: bool = True,
    include_citations: bool = False,
) -> List[Dict[str, Any]]:
    """Search PubMed and return rich metadata.

    Returns a list of dicts with keys: pmid, title, abstract, journal, year,
    authors, doi, pmcid, mesh_terms (optional), citations (optional: counts + ids).
    """
    if Entrez is None:
        raise ImportError("biopython is required: pip install biopython")

    # ESearch: get IDs
    handle = _entrez_call("esearch", db="pubmed", term=query, retmax=max_results)
    search = Entrez.read(handle)
    handle.close()
    id_list = search.get("IdList", [])
    if not id_list:
        return []

    # EFetch XML for rich fields
    handle = _entrez_call("efetch", db="pubmed", id=",".join(id_list), retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    def _extract_year(article_dict: Dict[str, Any]) -> Optional[int]:
        """Robust year extraction across common PubMed XML layouts."""
        med = article_dict.get("MedlineCitation", {}) or {}
        art = med.get("Article", {}) or {}

        # 1) ArticleDate (often for online-first)
        try:
            ad = art.get("ArticleDate")
            if isinstance(ad, list) and ad:
                y = ad[0].get("Year")
                if y:
                    return int(y)
        except Exception:
            pass

        # 2) Journal PubDate Year
        try:
            ji = (art.get("Journal") or {}).get("JournalIssue") or {}
            pd = ji.get("PubDate") or {}
            y = pd.get("Year")
            if y:
                return int(y)
            # 2b) MedlineDate like "2003 Jan-Feb" or "1998 Spring"
            md = pd.get("MedlineDate")
            if md:
                m = re.search(r"(\d{4})", str(md))
                if m:
                    return int(m.group(1))
        except Exception:
            pass

        # 3) MedlineCitation Date* fallbacks
        for key in ("DateCompleted", "DateRevised", "DateCreated"):
            try:
                y = (med.get(key) or {}).get("Year")
                if y:
                    return int(y)
            except Exception:
                continue

        # 4) PubmedData History (PubStatus timestamps)
        try:
            hist = (article_dict.get("PubmedData") or {}).get("History") or []
            # Prefer 'pubmed'/'entrez'/'medline' statuses if available
            preferred_order = ("pubmed", "entrez", "medline", "received", "accepted", "revised", "aheadofprint")
            # Build a map of status->year
            status_years = {}
            for d in hist:
                status = str(d.get("PubStatus", "")).lower()
                y = d.get("Year")
                if y:
                    status_years[status] = int(y)
            for s in preferred_order:
                if s in status_years:
                    return status_years[s]
            # Otherwise return any year present
            if status_years:
                return sorted(status_years.values())[0]
        except Exception:
            pass

        return None

    articles: List[Dict[str, Any]] = []
    for article in records.get("PubmedArticle", []):
        med = article.get("MedlineCitation", {})
        art = med.get("Article", {})

        pmid = str(med.get("PMID", ""))
        title = art.get("ArticleTitle", "")
        abstract = " ".join(
            [p for p in (art.get("Abstract", {}) or {}).get("AbstractText", [])]
        )
        journal = ((art.get("Journal", {}) or {}).get("Title") or "")

        year = _extract_year(article)

        # Authors
        authors = []
        for au in (art.get("AuthorList") or []):
            last = au.get("LastName") or ""
            fore = au.get("ForeName") or ""
            collective = au.get("CollectiveName")
            name = collective or (f"{fore} {last}".strip())
            if name:
                authors.append(name)

        # Identifiers
        doi = None
        pmcid = None
        for iden in (art.get("ELocationID") or []):
            if getattr(iden, "attributes", {}).get("EIdType", "").lower() == "doi":
                doi = str(iden)
        for iden in (med.get("ArticleIdList") or []):
            if getattr(iden, "attributes", {}).get("IdType") == "doi":
                doi = str(iden)
            if getattr(iden, "attributes", {}).get("IdType") == "pmc":
                pmcid = str(iden)

        rec: Dict[str, Any] = {
            "pmid": pmid,
            "title": _clean_whitespace(str(title)),
            "abstract": _clean_whitespace(str(abstract)),
            "journal": _clean_whitespace(str(journal)),
            "year": year,
            "authors": authors,
            "doi": doi,
            "pmcid": pmcid,
        }

        if include_mesh:
            mesh_terms = []
            for mh in (med.get("MeshHeadingList") or []):
                desc = (mh.get("DescriptorName") or "")
                if desc:
                    mesh_terms.append(str(desc))
            rec["mesh_terms"] = mesh_terms

        articles.append(rec)

    if include_citations and id_list:
        # One ELink call to fetch "cited in" (who cites this paper)
        try:
            link = _entrez_call(
                "elink",
                dbfrom="pubmed",
                id=",".join(id_list),
                linkname="pubmed_pubmed_citedin",
            )
            link_rec = Entrez.read(link)
            link.close()
            cited_map: Dict[str, List[str]] = {}
            for item in link_rec:
                src = None
                if item.get("IdList"):
                    # First Id in IdList is the source
                    src_ids = [str(x) for x in item.get("IdList") if str(x).isdigit()]
                    src = src_ids[0] if src_ids else None
                tgt = []
                for ldb in item.get("LinkSetDb", []):
                    if ldb.get("Link"):
                        tgt.extend([str(x["Id"]) for x in ldb["Link"] if x.get("Id")])
                if src:
                    cited_map[src] = tgt
            for rec in articles:
                pmid = rec["pmid"]
                ids = cited_map.get(pmid, [])
                rec["citations"] = {"count": len(ids), "pmids": ids}
        except Exception:
            # Fail soft; citation metadata is optional
            pass

    return articles



def query_scholar(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Best‑effort Google Scholar scrape (fragile by nature).

    Returns list of {title, url, snippet}. If the layout changes, returns an
    error entry instead of raising.
    """
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 and lxml are required for query_scholar")

    url = "https://scholar.google.com/scholar"
    params = {"q": query, "hl": "en"}
    try:
        r = _http_get(url, params=params)
        soup = BeautifulSoup(r.text, "lxml")
        out: List[Dict[str, Any]] = []
        for card in soup.select("div.gs_r.gs_or.gs_scl"):
            h3 = card.select_one("h3.gs_rt a")
            snip = card.select_one("div.gs_rs")
            if h3:
                out.append(
                    {
                        "title": h3.get_text(strip=True),
                        "url": h3.get("href"),
                        "snippet": (snip.get_text(" ", strip=True) if snip else ""),
                    }
                )
            if len(out) >= max_results:
                break
        return out or [{"error": "No results parsed (layout may have changed)."}]
    except Exception as e:
        return [{"error": f"Scholar fetch failed: {e}"}]


def query_arxiv(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Search arXiv for preprints. Uses the `arxiv` package if available.

    Returns list of {title, authors, summary, pdf_url, published}.
    """
    if arxiv_py is None:
        raise ImportError("arxiv package not installed. pip install arxiv")
    _limiter.wait("arxiv")
    search = arxiv_py.Search(query=query, max_results=max_results,
                             sort_by=arxiv_py.SortCriterion.Relevance)
    client = getattr(arxiv_py, "Client", None)
    results_iter = client().results(search) if client else search.results()
    out = []
    for r in results_iter:
        published = getattr(r, "published", None)
        out.append({
            "title": r.title,
            "authors": [a.name for a in r.authors],
            "summary": _clean_whitespace(r.summary or ""),
            "pdf_url": r.pdf_url,
            "published": published.isoformat() if hasattr(published, "isoformat") else str(published),
        })
    return out


# ---------------------------------------------------------------------------
# 2) Evidence extraction
# ---------------------------------------------------------------------------

def _find_doi(url: str):
    return re.findall(r'\b10\.\d{4,9}/[-.;()/:\w]+', url)

def extract_pdf_content(doi_or_url: str, *, max_pages: Optional[int] = None) -> Dict[str, Any]:
    """Download a PDF and return extracted plain text.

    Returns {text, meta:{pages, bytes}}. Attempts extraction even if the server
    omits a PDF content-type. Requires PyPDF2.
    """
    if PyPDF2 is None:
        raise ImportError("PyPDF2 is required: pip install pypdf2")

    try:
        email = os.environ["ENTREZ_EMAIL"]
    except Exception:
        email = NCBI_EMAIL

    if _find_doi(doi_or_url):
        doi = _find_doi(doi_or_url)[0]
        raw = None
        resp = _http_get(f"https://api.unpaywall.org/v2/{doi}?email={email}", stream=True)
        fields = resp.json()
        for loc in fields['oa_locations']:
            if 'url_for_pdf' in loc:
                try:
                    pdfresp = _http_get(loc["url_for_pdf"], stream=True)
                    raw = pdfresp.content
                    if raw:
                        break
                except Exception:
                    continue
    else:
        resp = _http_get(doi_or_url, stream=True)
        raw = resp.content
    with BytesIO(raw) as fh:
        reader = PyPDF2.PdfReader(fh)
        n_pages = len(reader.pages)
        pages_to_read = range(n_pages if max_pages is None else min(max_pages, n_pages))
        chunks: List[str] = []
        for i in pages_to_read:
            try:
                pg = reader.pages[i]
                txt = pg.extract_text() or ""
                chunks.append(txt)
            except Exception:
                continue
        text = _clean_whitespace("\n".join(chunks))
    result = {"text": text, "meta": {"pages": n_pages, "bytes": len(raw)}}
    return result


def extract_url_content(url: str) -> Dict[str, Any]:
    """Fetch a webpage and strip boilerplate.

    If readability-lxml is installed, use it; otherwise fall back to BeautifulSoup
    heuristics. Returns {text, title}.
    """
    if BeautifulSoup is None:
        raise ImportError("beautifulsoup4 and lxml are required for extract_url_content")

    r = _http_get(url)
    html = r.text
    title = None
    if ReadabilityDoc is not None:
        try:
            doc = ReadabilityDoc(html)
            title = doc.short_title()
            cleaned_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(cleaned_html, "lxml")
            text = _clean_whitespace(soup.get_text(" "))
            return {"text": text, "title": title}
        except Exception:
            pass

    soup = BeautifulSoup(html, "lxml")
    title = (soup.title.get_text(strip=True) if soup.title else None)
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    body = soup.body or soup
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all(["p", "h1", "h2", "h3", "li"]) if p.get_text(strip=True)]
    return {"text": _clean_whitespace("\n".join(paragraphs)), "title": title}


# ---------------------------------------------------------------------------
# 3) Clinical‑genetics reasoning
# ---------------------------------------------------------------------------

def query_clinvar(variant: str) -> Dict[str, Any]:
    """Retrieve ClinVar details for a variant query (rsID or HGVS).

    Fields: title, clinical_significance, review_status, conditions, gene,
    accessions (RCV/VCV), hgvs, pubmed_pmids, allele_frequencies (if available).
    Uses NCBI E-utilities (ClinVar XML) and attempts allele frequencies via
    Variation Services (for rsIDs).
    """
    if Entrez is None:
        raise ImportError("biopython is required: pip install biopython")

    # ESearch in ClinVar
    s = _entrez_call("esearch", db="clinvar", term=variant)
    s_rec = Entrez.read(s)
    s.close()
    ids = s_rec.get("IdList", [])
    if not ids:
        return {"error": "Variant not found in ClinVar."}

    # Use first match for simplicity (minimal action space)
    uid = ids[0]
    f = _entrez_call("efetch", db="clinvar", id=uid, retmode="xml")
    x = Entrez.read(f)
    f.close()

    # ClinVar XML is verbose; pull common fields
    # Handle different XML response formats
    if hasattr(x, 'get'):
        doc = x.get("ClinVarSet", x).get("ClinVarSet", x)  # tolerate shape
    else:
        # Handle ListElement responses - use esummary instead for cleaner parsing
        doc = None
    
    title = None
    significance = None
    review_status = None
    conditions: List[str] = []
    gene = None
    hgvs_list: List[str] = []
    pubmed_pmids: List[str] = []
    accessions: Dict[str, List[str]] = {"RCV": [], "VCV": []}

    # Walk nested structure defensively - use esummary with validation disabled
    try:
        # Use esummary for cleaner structured data
        summ = _entrez_call("esummary", db="clinvar", id=uid)
        summ_rec = Entrez.read(summ, validate=False)  # Disable DTD validation
        summ.close()
        
        if "DocumentSummarySet" in summ_rec:
            ds = summ_rec["DocumentSummarySet"]["DocumentSummary"][0]
            
            # Extract title
            title = ds.get("title")
            
            # Extract clinical significance - try multiple fields due to ClinVar schema changes
            significance = None
            review_status = None
            
            # Try germline_classification (newer format)
            germline = ds.get("germline_classification")
            if germline and isinstance(germline, dict):
                significance = germline.get("description") or germline.get("last_evaluated")
                review_status = germline.get("review_status")
            
            # Try clinical_significance (older format)
            if not significance:
                clin_sig = ds.get("clinical_significance")
                if isinstance(clin_sig, dict):
                    significance = clin_sig.get("description")
                    review_status = clin_sig.get("review_status")
                elif isinstance(clin_sig, str):
                    significance = clin_sig
            
            # Fallback to direct review_status field
            if not review_status:
                review_status = ds.get("review_status")
            
            # Extract conditions/traits - check both trait_set and germline_classification.trait_set
            conditions = []
            
            # Check top-level trait_set
            trait_set = ds.get("trait_set", [])
            if isinstance(trait_set, list):
                conditions.extend([t.get("trait_name") for t in trait_set if isinstance(t, dict) and t.get("trait_name")])
            
            # Check germline_classification.trait_set (newer location)
            if germline and isinstance(germline, dict):
                germline_traits = germline.get("trait_set", [])
                if isinstance(germline_traits, list):
                    conditions.extend([t.get("trait_name") for t in germline_traits if isinstance(t, dict) and t.get("trait_name")])
            
            # Remove duplicates
            conditions = list(set(filter(None, conditions)))
            
            # Extract genes
            genes = ds.get("genes", [])
            if isinstance(genes, list) and genes:
                gene_info = genes[0]
                if isinstance(gene_info, dict):
                    gene = gene_info.get("symbol")
                else:
                    gene = None
            else:
                gene = None
            
            # Extract accession
            acc = ds.get("accession")
            if acc:
                if str(acc).startswith("RCV"):
                    accessions["RCV"].append(str(acc))
                elif str(acc).startswith("VCV"):
                    accessions["VCV"].append(str(acc))
            
            # Extract HGVS expressions
            hgvs = ds.get("hgvs_expressions", {})
            if isinstance(hgvs, dict):
                for v in hgvs.values():
                    if isinstance(v, list):
                        hgvs_list.extend([str(i) for i in v])
                    elif v:
                        hgvs_list.append(str(v))
                        
    except Exception as e:
        # If esummary fails, significance will remain None
        pass

    # PubMed links via ELink
    try:
        el = _entrez_call("elink", dbfrom="clinvar", id=uid, linkname="clinvar_pubmed")
        elr = Entrez.read(el)
        el.close()
        for ls in elr:
            for db in ls.get("LinkSetDb", []):
                if db.get("DbTo") == "pubmed":
                    pubmed_pmids = [str(l["Id"]) for l in db.get("Link", [])]
    except Exception:
        pass

    # Extract allele frequencies from ClinVar variation_set (preferred) or Variation Services
    allele_freqs: Dict[str, float] = {}
    
    # Try extracting from ClinVar esummary first (more reliable)
    try:
        if "DocumentSummarySet" in summ_rec:
            ds = summ_rec["DocumentSummarySet"]["DocumentSummary"][0]
            variation_set = ds.get("variation_set", [])
            
            if isinstance(variation_set, list):
                for variation in variation_set:
                    if isinstance(variation, dict):
                        allele_freq_set = variation.get("allele_freq_set", [])
                        if isinstance(allele_freq_set, list):
                            for freq_data in allele_freq_set:
                                if isinstance(freq_data, dict):
                                    source = freq_data.get("source", "")
                                    value = freq_data.get("value", "")
                                    if value:
                                        try:
                                            freq_val = float(value)
                                            # Keep highest frequency across all sources/variants
                                            key = f"{source}_{variation.get('variation_name', 'unknown')}"
                                            allele_freqs[key] = max(freq_val, allele_freqs.get(key, 0.0))
                                        except (ValueError, TypeError):
                                            pass
    except Exception:
        pass
    
    # Fallback to Variation Services if no frequencies found and we have an rsID
    if not allele_freqs and re.match(r"^rs\d+$", variant, re.IGNORECASE):
        try:
            vs = _http_get(f"https://api.ncbi.nlm.nih.gov/variation/v0/refsnp/{variant.lower()[2:]}")
            j = vs.json()
            # Aggregate observed allele frequencies if present
            for ann in j.get("primary_snapshot_data", {}).get("allele_annotations", []):
                for fset in ann.get("frequency", []):
                    for obs in fset.get("study_results", []):
                        alt = obs.get("allele" ) or obs.get("x_allele")
                        if "allele_freq" in obs:
                            freq_val = float(obs.get("allele_freq") or 0.0)
                            allele_freqs[f"vs_{alt}"] = max(freq_val, allele_freqs.get(f"vs_{alt}", 0.0))
        except Exception:
            pass

    return {
        "query": variant,
        "title": title,
        "clinical_significance": significance,
        "review_status": review_status,
        "conditions": conditions,
        "gene": gene,
        "accessions": accessions,
        "hgvs": sorted(set(hgvs_list)),
        "pubmed_pmids": pubmed_pmids,
        "allele_frequencies": allele_freqs or None,
    }


def query_omim(term: str, *, include: str = "geneMap,clinicalSynopsis,allelicVariant") -> Dict[str, Any]:
    """Query OMIM live API by MIM number or gene symbol.

    Requires OMIM_API_KEY. Returns a compact dict with gene symbol (if any),
    phenotype titles, inheritance, clinical synopsis highlights, allelic variants
    (IDs + simple names), and molecular mechanism if available.
    """
    api_key = os.getenv("OMIM_API_KEY")
    if not api_key:
        return {"error": "OMIM_API_KEY not set. Set it to use the live OMIM API."}

    base = "https://api.omim.org/api/entry"
    params = {"format": "json", "include": include, "apiKey": api_key}
    # Detect numeric MIM vs symbol/text
    if re.fullmatch(r"\d{6}", term):
        params["mimNumber"] = term
    else:
        params["search"] = term

    resp = _http_get(base, params=params)
    j = resp.json()
    entries = (j.get("omim", {}).get("entryList") or [])
    if not entries:
        return {"term": term, "results": []}

    results: List[Dict[str, Any]] = []
    for wrap in entries:
        e = wrap.get("entry", {})
        gene_symbols = []
        if e.get("geneMap") and e["geneMap"].get("geneMapMethods"):
            # prefer approved symbol if present
            sym = e["geneMap"].get("geneSymbols") or ""
            gene_symbols = [s.strip() for s in sym.split(",") if s.strip()]

        phenotypes = []
        for p in (e.get("geneMap", {}) or {}).get("phenotypeMapList", []) or []:
            phen = p.get("phenotypeMap", {})
            phenotypes.append(
                {
                    "phenotype": phen.get("phenotype") or phen.get("phenotypeMimNumber"),
                    "inheritance": phen.get("inheritance"),
                    "mim": phen.get("phenotypeMimNumber"),
                }
            )

        clin_syn = e.get("clinicalSynopsis") or {}
        mechanism = (e.get("textSectionList") or [{}])[0].get("textSection", {}).get("textSectionContent")

        allelic_variants = []
        for av in (e.get("allelicVariantList") or []):
            avx = av.get("allelicVariant", {})
            allelic_variants.append(
                {
                    "mim": avx.get("allelicVariantMimNumber"),
                    "name": avx.get("allelicVariantName"),
                    "dbsnps": avx.get("dbSnps"),
                }
            )

        results.append(
            {
                "mimNumber": e.get("mimNumber"),
                "titles": e.get("titles", {}),
                "gene_symbols": gene_symbols,
                "phenotypes": phenotypes,
                "inheritance": clin_syn.get("inheritance"),
                "clinicalSynopsis": {k: clin_syn.get(k) for k in ["inheritance", "phenotype", "molecularBasis"] if k in clin_syn},
                "molecular_mechanism": mechanism,
                "allelic_variants": allelic_variants,
            }
        )

    return {"term": term, "results": results}


# ---------------------------------------------------------------------------
# 4) Sequence analytics (BLAST)
# ---------------------------------------------------------------------------

def blast_sequence(
    sequence: str,
    *,
    program: str = "blastn",
    database: Optional[str] = None,
    hitlist_size: int = 10,
) -> Dict[str, Any]:
    """Submit a BLAST job to NCBI and return top hits (structured).

    For DNA use program="blastn" (default db="nt"); for protein use
    program="blastp" (default db="nr"). Returns {query_length, hits:[...]}
    where each hit contains id, title, length, hsps:[{evalue, score, identities,
    align_len, pct_identity}].
    """
    if NCBIWWW is None or NCBIXML is None:
        raise ImportError("biopython is required for BLAST: pip install biopython")

    if not sequence or not re.fullmatch(r"[A-Za-z*\-\.\?]+", sequence):
        return {"error": "Invalid sequence string."}

    if database is None:
        database = "nt" if program.lower() == "blastn" else "nr"

    _limiter.wait("ncbi")
    result_handle = NCBIWWW.qblast(program, database, sequence, hitlist_size=hitlist_size)
    record = NCBIXML.read(result_handle)

    hits = []
    for aln in record.alignments[:hitlist_size]:
        hsp_summaries = []
        for hsp in aln.hsps:
            pct = (100.0 * hsp.identities / hsp.align_length) if getattr(hsp, "align_length", 0) else None
            hsp_summaries.append(
                {
                    "evalue": getattr(hsp, "expect", None),
                    "score": getattr(hsp, "score", None),
                    "identities": getattr(hsp, "identities", None),
                    "align_len": getattr(hsp, "align_length", None),
                    "pct_identity": pct,
                }
            )
        hits.append(
            {
                "hit_id": getattr(aln, "hit_id", None),
                "title": getattr(aln, "title", None),
                "length": getattr(aln, "length", None),
                "hsps": hsp_summaries,
            }
        )

    return {"query_length": getattr(record, "query_length", None), "hits": hits}


# ---------------------------------------------------------------------------
# 5) ClinicalTrials.gov (live endpoint)
# ---------------------------------------------------------------------------

def query_clinicaltrials(
    expr: str,
    *,
    max_results: int = 50,
    fields: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Query ClinicalTrials.gov v2 API and project common study fields."""
    if fields is None:
        fields = [
            "NCTId",
            "BriefTitle",
            "OverallStatus",
            "StudyType",
            "Condition",
            "InterventionName",
            "PrimaryOutcomeMeasure",
            "EnrollmentCount",
            "StartDate",
            "CompletionDate",
            "Phase",
            "LocationCountry",
            "LeadSponsorName",
        ]

    base_url = "https://clinicaltrials.gov/api/v2/studies"
    page_size = max(1, min(max_results, 100))  # v2 returns paged results
    params = {
        "query.term": expr,         # v1 `expr` -> v2 `query.term`
        "format": "json",
        "pageSize": page_size,
        "countTotal": "true",       # include totalCount in response
    }

    def _project(s: Dict[str, Any]) -> Dict[str, Any]:
        ps = s.get("protocolSection", {})
        idm = ps.get("identificationModule", {}) or {}
        sm  = ps.get("statusModule", {}) or {}
        dm  = ps.get("designModule", {}) or {}
        cm  = ps.get("conditionsModule", {}) or {}
        aim = ps.get("armsInterventionsModule", {}) or {}
        om  = ps.get("outcomesModule", {}) or {}
        clm = ps.get("contactsLocationsModule", {}) or {}
        scm = ps.get("sponsorCollaboratorsModule", {}) or {}

        # helpers
        def _names(items, *keys):
            out = []
            for it in (items or []):
                if isinstance(it, dict):
                    for k in keys:
                        if it.get(k):
                            out.append(it[k])
                            break
            return out

        rec = {}
        for f in fields:
            if f == "NCTId":
                rec[f] = idm.get("nctId")
            elif f == "BriefTitle":
                rec[f] = idm.get("briefTitle")
            elif f == "OverallStatus":
                rec[f] = sm.get("overallStatus")
            elif f == "StudyType":
                rec[f] = dm.get("studyType")
            elif f == "Condition":
                rec[f] = cm.get("conditions")  # list[str]
            elif f == "InterventionName":
                rec[f] = _names(aim.get("interventions", []), "name", "interventionName")
            elif f == "PrimaryOutcomeMeasure":
                # v2 commonly uses 'name'; some records may still have 'measure'
                rec[f] = _names(om.get("primaryOutcomes", []), "name", "measure")
            elif f == "EnrollmentCount":
                rec[f] = (dm.get("enrollmentInfo") or {}).get("count")
            elif f == "StartDate":
                rec[f] = (sm.get("startDateStruct") or {}).get("date")
            elif f == "CompletionDate":
                rec[f] = (sm.get("completionDateStruct") or {}).get("date")
            elif f == "Phase":
                rec[f] = dm.get("phases") or dm.get("phase")
            elif f == "LocationCountry":
                rec[f] = sorted({
                    loc.get("country")
                    for loc in (clm.get("locations") or [])
                    if isinstance(loc, dict) and loc.get("country")
                })
            elif f == "LeadSponsorName":
                lead = scm.get("leadSponsor") or (scm.get("sponsors") or {}).get("leadSponsor")
                rec[f] = (lead or {}).get("name") or (lead or {}).get("fullName") or (lead or {}).get("agency")
            else:
                rec[f] = None
        return rec

    studies_out: List[Dict[str, Any]] = []
    next_token = None
    total_count = None

    while len(studies_out) < max_results:
        if next_token:
            params["pageToken"] = next_token
        r = _http_get(base_url, params=params)
        r.raise_for_status()
        payload = r.json() or {}
        if total_count is None:
            total_count = payload.get("totalCount")  # present when countTotal=true
        for s in payload.get("studies", []):
            studies_out.append(_project(s))
            if len(studies_out) >= max_results:
                break
        next_token = payload.get("nextPageToken")
        if not next_token:
            break

    return {"count": total_count, "studies": studies_out}



# ---------------------------------------------------------------------------
# 6) Persistent REPL (Python, with optional Bash/R via shebang)
# ---------------------------------------------------------------------------

_persistent_namespace: Dict[str, Any] = {}


def run_python_repl(code: str) -> Dict[str, Any]:
    """Execute Python code in a persistent namespace.

    Also supports **shebang** prefixes for quick Bash/R ops:
        #!bash\n<shell commands>
        #!r\n<single‑file R script>

    Returns {stdout, error}. CAUTION: This executes arbitrary code.
    """
    code = code.strip()

    # Strip Markdown fences if present
    if code.startswith("```"):
        code = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", code).strip()

    # --- Bash mode ---
    if code.startswith("#!bash"):
        script = code.split("\n", 1)[1] if "\n" in code else ""
        return _run_shell(script)

    # --- R mode ---
    if re.match(r"^#!r(\b|\n)", code, flags=re.IGNORECASE):
        script = code.split("\n", 1)[1] if "\n" in code else ""
        return _run_r(script)

    # --- Python mode ---
    global _persistent_namespace
    stdout_buf = StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = stdout_buf
    sys.stderr = stdout_buf
    error: Optional[str] = None
    try:
        exec(code, _persistent_namespace)
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    return {"stdout": stdout_buf.getvalue(), "error": error}


def _run_shell(script: str) -> Dict[str, Any]:
    import subprocess, tempfile

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run(["/bin/bash", path], capture_output=True, text=True)
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return {"stdout": out, "error": None if proc.returncode == 0 else f"Exit {proc.returncode}"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _run_r(script: str) -> Dict[str, Any]:
    import subprocess, tempfile

    if shutil.which("Rscript") is None:
        return {"stdout": "", "error": "Rscript not found in PATH."}
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".R") as tf:
        tf.write(script)
        path = tf.name
    try:
        proc = subprocess.run(["Rscript", path], capture_output=True, text=True)
        out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return {"stdout": out, "error": None if proc.returncode == 0 else f"Exit {proc.returncode}"}
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 7) Local knowledge bases (CSV/TSV snapshots)
# ---------------------------------------------------------------------------

def query_local_clinvar(
    gene_symbol: str,
    tsv_path: str = "./data/clinvar_snapshot.tsv",
    *,
    cols: Optional[List[str]] = None,
    max_rows: int = 100,
) -> Dict[str, Any]:
    """Filter a local ClinVar TSV snapshot by gene symbol.

    Returns {rows:[...], columns:[...]}. Auto‑detects common column names.
    """
    if not os.path.exists(tsv_path):
        return {"error": f"ClinVar snapshot not found at '{tsv_path}'"}

    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    gene_cols = [c for c in df.columns if c.lower() in {"gene(s)", "genes", "genesymbol", "gene_symbol"}]
    if not gene_cols:
        return {"error": "Could not find gene symbol column in TSV."}
    gene_col = gene_cols[0]

    mask = df[gene_col].astype(str).str.contains(fr"\b{re.escape(gene_symbol)}\b", case=False, na=False)
    sub = df.loc[mask]
    if sub.empty:
        return {"rows": [], "columns": list(df.columns)}

    if cols is None:
        cols = [c for c in [
            "Name",
            gene_col,
            "Clinical significance (Last reviewed)",
            "ClinicalSignificance",
            "Review status",
            "ReviewStatus",
            "RCVaccession",
        ] if c in sub.columns]
        cols = cols or list(sub.columns)  # fallback

    out = sub[cols].head(max_rows)
    return {"rows": out.to_dict(orient="records"), "columns": list(out.columns)}


def query_local_omim(
    query: str,
    csv_path: str = "./data/omim_catalog.csv",
    *,
    max_rows: int = 200,
) -> Dict[str, Any]:
    """Filter a local OMIM CSV by gene symbol or phenotype keyword.

    Expects columns like 'Gene Symbols' and 'Phenotypes'. Returns {rows, columns}.
    """
    if not os.path.exists(csv_path):
        return {"error": f"OMIM catalog not found at '{csv_path}'"}

    df = pd.read_csv(csv_path, low_memory=False)
    cand_cols = {c.lower(): c for c in df.columns}
    gs = cand_cols.get("gene symbols") or cand_cols.get("gene_symbols")
    ph = cand_cols.get("phenotypes") or cand_cols.get("phenotype")
    if not gs or not ph:
        return {"error": "CSV must contain 'Gene Symbols' and 'Phenotypes' columns."}

    m = df[gs].astype(str).str.contains(query, case=False, na=False) | df[ph].astype(str).str.contains(query, case=False, na=False)
    sub = df.loc[m].head(max_rows)
    return {"rows": sub.to_dict(orient="records"), "columns": list(sub.columns)}


def query_local_faers(
    drug_name: str,
    csv_dir: str = "./data/faers",
    *,
    top_n: int = 20,
) -> Dict[str, Any]:
    """Summarize top MedDRA Preferred Terms for a drug from local FAERS CSVs.

    Expects per‑year CSVs with at least columns: 'drugname' and 'pt'. Returns
    {top_reactions:[{pt, count}], total_reports}.
    """
    if not os.path.isdir(csv_dir):
        return {"error": f"FAERS directory not found at '{csv_dir}'"}

    files = sorted(glob.glob(os.path.join(csv_dir, "faers_*.csv")))
    if not files:
        return {"error": f"No FAERS CSV files found in '{csv_dir}'"}

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, usecols=["drugname", "pt"], on_bad_lines="skip", low_memory=False))
        except Exception:
            continue
    if not frames:
        return {"error": "No FAERS data could be read (check schema)."}

    df = pd.concat(frames, ignore_index=True)
    mask = df["drugname"].astype(str).str.contains(drug_name, case=False, na=False)
    sub = df.loc[mask]
    total = int(len(sub))
    if total == 0:
        return {"top_reactions": [], "total_reports": 0}

    counts = (
        sub["pt"].astype(str).str.strip().value_counts().head(top_n).reset_index()
    )
    counts.columns = ["pt", "count"]
    return {
        "top_reactions": counts.to_dict(orient="records"),
        "total_reports": total,
    }

if __name__ == "__main__":
    pm = query_pubmed("Parkinsons")
    print(pm)
    print(query_clinicaltrials("Alzheimer's"))
    print(1)