"""Answer generation grounded only in the retrieved papers."""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a medical AI assistant that provides accurate information based on published "
    "medical research.\n"
    "Answer the question using only the papers below. Each excerpt starts with a header "
    "[PMID | first author | year]. Cite claims in the form (PMID: <number>, <first author> et al., <year>), "
    "copying the PMID, author and year exactly from the header of the excerpt that supports the claim.\n"
    "Never cite a PMID or author that does not appear in a header below, and never combine findings "
    "from different papers into one citation.\n"
    "If the papers do not contain enough information, say \"Insufficient evidence\" and explain why.\n\n"
    "Papers:\n{context}\n\nQuestion:\n{question}\n"
)

PMID_PATTERN = re.compile(r"PMID[:\s]*(\d{4,9})")


def first_author(authors: str) -> str:
    """'Smith J, Lee K' -> 'Smith'."""
    first = (authors or "").split(",")[0].strip()
    return first.split(" ")[0] if first else "Unknown"


def format_context(docs) -> str:
    """Prefix every chunk with its paper's header.

    Chunks after the first one of an abstract carry no PMID in their text, so
    without this header the model cannot tell which paper an excerpt came from
    and may invent a citation.
    """
    blocks = []
    for d in docs:
        m = d.metadata
        header = f"[PMID: {m.get('pmid', '')} | {first_author(m.get('authors', ''))} | {m.get('year', '')}]"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n".join(blocks)


def cited_pmids(answer: str) -> List[str]:
    """PMIDs mentioned in the answer, in order of first appearance."""
    return list(dict.fromkeys(PMID_PATTERN.findall(answer)))


def invalid_citations(answer: str, sources: List[Dict]) -> List[str]:
    """PMIDs the answer cites that were not among the retrieved sources (fabricated citations)."""
    allowed = {s["pmid"] for s in sources}
    return [p for p in cited_pmids(answer) if p not in allowed]


def unique_sources(docs) -> List[Dict]:
    """One source per PMID, in retrieval order."""
    seen, sources = set(), []
    for d in docs:
        pmid = d.metadata.get("pmid", "")
        if pmid and pmid not in seen:
            seen.add(pmid)
            sources.append({k: d.metadata.get(k, "") for k in
                            ("pmid", "title", "authors", "journal", "year", "abstract")})
    return sources


def generate_answer(question: str, docs, llm) -> Tuple[str, List[Dict]]:
    if not docs:
        return "Insufficient evidence: no relevant PubMed papers were retrieved for this question.", []
    answer = (ANSWER_PROMPT | llm | StrOutputParser()).invoke(
        {"context": format_context(docs), "question": question})
    return answer, unique_sources(docs)