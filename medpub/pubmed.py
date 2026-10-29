"""PubMed search through the NCBI Entrez API."""
from __future__ import annotations

from typing import Dict, List

from Bio import Entrez, Medline


class PubMedError(RuntimeError):
    """Raised when the Entrez API cannot be reached or returns an error."""


def record_to_paper(record: dict) -> Dict:
    """Map one MEDLINE record to the paper dict used by the rest of the pipeline."""
    doi = next((ref for ref in record.get("AID", []) if ref.endswith("[doi]")), "")
    paper = {
        "pmid": record.get("PMID", ""),
        "title": record.get("TI", "No title"),
        "abstract": record.get("AB", ""),
        "authors": ", ".join(record.get("AU", ["No author"])),
        "journal": record.get("TA", "No journal"),
        "year": record.get("DP", "").split()[0] if record.get("DP") else "",
        "doi": doi.replace("[doi]", "").strip(),
        "types": record.get("PT", []),
    }
    return paper


def full_text(paper: Dict) -> str:
    """Text that gets chunked and embedded: metadata first, then the abstract."""
    return (
        f"Title: {paper['title']}\n"
        f"Authors: {paper['authors']}\n"
        f"Journal: {paper['journal']}\n"
        f"Year: {paper['year']}\n"
        f"PMID: {paper['pmid']}\n"
        f"DOI: {paper['doi']}\n"
        f"Abstract: {paper['abstract'] or 'No abstract available'}"
    )


def _esearch_ids(term: str, max_results: int) -> List[str]:
    handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results, sort="relevance")
    result = Entrez.read(handle)
    handle.close()
    return list(result["IdList"])


def _efetch_records(ids: List[str]) -> List[dict]:
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()
    return records


def _efetch_abstract(pmid: str) -> str:
    handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="text")
    text = handle.read()
    handle.close()
    return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text


def search_pubmed(query: str, email: str, max_results: int = 10,
                  start_year: int = 2020, end_year: int = 2026) -> List[Dict]:
    """Most relevant papers for `query` published between start_year and end_year."""
    Entrez.email = email or "anonymous@example.com"
    term = f"{query} AND {start_year}:{end_year}[dp]"
    try:
        ids = _esearch_ids(term, max_results)
        if not ids:
            return []
        records = _efetch_records(ids)
    except Exception as exc:  # network errors, HTTP errors, parse errors
        raise PubMedError(f"PubMed search failed: {exc}") from exc

    papers = []
    for record in records:
        paper = record_to_paper(record)
        if not paper["abstract"].strip():
            try:
                paper["abstract"] = _efetch_abstract(paper["pmid"]).strip()
            except Exception:
                pass  # keep the paper; it is still citable by title
        paper["full_text"] = full_text(paper)
        papers.append(paper)
    return papers
