"""Query strategies: original, Multi-Query, HyDE and Step-Back."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from .config import QUERY_METHODS


def unique_union(document_lists: list) -> list:
    """Flatten lists of documents and drop duplicates (same text from the same paper)."""
    seen, out = set(), []
    for docs in document_lists:
        for doc in docs:
            key = (doc.metadata.get("pmid", ""), doc.page_content)
            if key not in seen:
                seen.add(key)
                out.append(doc)
    return out


def query_multiquery(question, retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "You are an AI language model assistant. Generate five different versions of the given "
        "question to retrieve relevant documents from a vector database. Provide these alternative "
        "questions separated by newlines.\nOriginal question: {question}"
    )
    queries = (prompt | llm | StrOutputParser()).invoke({"question": question})
    variants = [q.strip() for q in queries.split("\n") if q.strip()]
    return unique_union([retriever.invoke(q) for q in [question, *variants]])


def query_stepback(question, retriever, llm):
    examples = [
        {"input": "Could the members of The Police perform lawful arrests?",
         "output": "What can the members of The Police do?"},
        {"input": "Jan Sindel's was born in what country?",
         "output": "What is Jan Sindel's personal history?"},
    ]
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")]),
        examples=examples,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at world knowledge. Step back and paraphrase a question "
                   "to a more generic step-back question."),
        few_shot,
        ("user", "{question}"),
    ])
    step_back = (prompt | llm | StrOutputParser()).invoke({"question": question})
    return unique_union([retriever.invoke(question), retriever.invoke(step_back)])


def query_hyde(question, retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        "Please write a scientific paper passage to answer the question.\nQuestion: {question}\nPassage:"
    )
    passage = (prompt | llm | StrOutputParser()).invoke({"question": question})
    return retriever.invoke(passage)


def run_query(question, method, retriever, llm):
    if method not in QUERY_METHODS:
        raise ValueError(f"Unknown query method: {method}")
    if method == "original":
        return retriever.invoke(question)
    if method == "multiquery":
        return query_multiquery(question, retriever, llm)
    if method == "hyde":
        return query_hyde(question, retriever, llm)
    return query_stepback(question, retriever, llm)
