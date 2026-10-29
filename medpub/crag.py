"""Corrective RAG graph: retrieve → grade → (rewrite → re-search) → generate.

Changes from the original notebook/app version:
- The rewritten query is used only for the second PubMed search; the answer is
  still written for the user's original question.
- Every dependency (LLMs, grader, search, indexing) is passed in, so the graph
  runs offline in tests with fakes.
"""
from __future__ import annotations

from typing import Callable, List, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .generation import generate_answer
from .queries import run_query


class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Document is relevant to the question, 'yes' or 'no'")


GRADE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. "
               "Give a binary score 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a question re-writer that converts an input question to a better version "
               "optimized for searching PubMed medical literature. Reply with the question only."),
    ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
])


def llm_grader(llm) -> Callable[[str, str], bool]:
    """Relevance judge: True if the LLM says the document is relevant."""
    chain = GRADE_PROMPT | llm.with_structured_output(GradeDocuments)
    return lambda question, document: chain.invoke(
        {"question": question, "document": document}).binary_score.strip().lower() == "yes"


class GraphState(TypedDict, total=False):
    question: str            # the user's question, never rewritten
    search_query: str        # query used for the corrective PubMed search
    query_method: str
    documents: list
    needs_research: bool
    generation: str
    sources: List[dict]


def build_crag_app(retriever, *, query_llm, answer_llm, rewrite_llm, grade: Callable[[str, str], bool],
                   research: Callable[[str], list]):
    """research(query) must return extra Documents for the corrective step."""

    rewriter = REWRITE_PROMPT | rewrite_llm | StrOutputParser()

    def retrieve(state: GraphState):
        docs = run_query(state["question"], state["query_method"], retriever, query_llm)
        return {"documents": docs}

    def grade_documents(state: GraphState):
        kept = [d for d in state["documents"] if grade(state["question"], d.page_content)]
        return {"documents": kept, "needs_research": len(kept) < len(state["documents"]) or not kept}

    def transform_query(state: GraphState):
        return {"search_query": rewriter.invoke({"question": state["question"]}).strip()}

    def research_again(state: GraphState):
        extra = research(state["search_query"])
        return {"documents": list(state["documents"]) + list(extra)}

    def generate(state: GraphState):
        answer, sources = generate_answer(state["question"], state["documents"], answer_llm)
        return {"generation": answer, "sources": sources}

    def route(state: GraphState):
        return "transform_query" if state.get("needs_research") else "generate"

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("transform_query", transform_query)
    graph.add_node("research_again", research_again)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route,
                                {"transform_query": "transform_query", "generate": "generate"})
    graph.add_edge("transform_query", "research_again")
    graph.add_edge("research_again", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
