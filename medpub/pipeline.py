"""One call from question to grounded answer: answer_question(params, settings)."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .config import Settings
from .crag import build_crag_app, llm_grader
from .indexing import build_faiss
from .pubmed import search_pubmed

TOP_K = 5


@dataclass
class AskParams:
    question: str
    query_method: str = "original"
    max_results: int = 10
    start_year: int = 2020
    end_year: int = field(default_factory=lambda: dt.date.today().year)
    model: str = "gpt-4o-mini"


@dataclass
class Components:
    """Everything that talks to the outside world, swappable in tests."""
    search: Callable[..., List[Dict]]
    embeddings: object
    query_llm: object
    answer_llm: object
    rewrite_llm: object
    grade: Callable[[str, str], bool]


def default_components(settings: Settings, model: str) -> Components:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    key = settings.openai_api_key
    helper = ChatOpenAI(model=settings.grader_model, temperature=0, api_key=key)
    return Components(
        search=search_pubmed,
        embeddings=OpenAIEmbeddings(api_key=key),
        query_llm=helper,
        answer_llm=ChatOpenAI(model=model, temperature=0, api_key=key),
        rewrite_llm=helper,
        grade=llm_grader(helper),
    )


def answer_question(params: AskParams, settings: Settings,
                    components: Optional[Components] = None) -> Dict:
    """Search PubMed, run the CRAG graph and return answer, sources and the node trace."""
    t0 = time.perf_counter()
    c = components or default_components(settings, params.model)

    def search(query: str) -> List[Dict]:
        return c.search(query=query, email=settings.entrez_email, max_results=params.max_results,
                        start_year=params.start_year, end_year=params.end_year)

    papers = search(params.question)
    if not papers:
        return {"answer": "Insufficient evidence: PubMed returned no papers for this question.",
                "sources": [], "nodes": [], "n_papers": 0,
                "latency_s": round(time.perf_counter() - t0, 2)}

    retriever = build_faiss(papers, c.embeddings).as_retriever(search_kwargs={"k": TOP_K})

    def research(query: str):
        extra = search(query)
        if not extra:
            return []
        return build_faiss(extra, c.embeddings).as_retriever(search_kwargs={"k": TOP_K}).invoke(query)

    app = build_crag_app(retriever, query_llm=c.query_llm, answer_llm=c.answer_llm,
                         rewrite_llm=c.rewrite_llm, grade=c.grade, research=research)

    nodes, final = [], {}
    for step in app.stream({"question": params.question, "query_method": params.query_method}):
        for name, update in step.items():
            nodes.append(name)
            final.update(update or {})

    return {"answer": final.get("generation", ""), "sources": final.get("sources", []),
            "nodes": nodes, "n_papers": len(papers),
            "latency_s": round(time.perf_counter() - t0, 2)}
