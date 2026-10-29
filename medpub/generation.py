"""Answer generation grounded only in the retrieved papers."""
from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a medical AI assistant that provides accurate information based on published "
    "medical research.\n"
    "Answer the question using only the provided papers. Always cite PMID, author(s), and year.\n"
    "If the papers do not contain enough information, say \"Insufficient evidence\" and explain why.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n"
)


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
    context = "\n\n".join(d.page_content for d in docs)
    answer = (ANSWER_PROMPT | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return answer, unique_sources(docs)
