"""Chunk paper text and index it in an in-memory FAISS store (one store per question)."""
from __future__ import annotations

from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
METADATA_KEYS = ("pmid", "title", "authors", "journal", "year", "abstract")


def chunk_papers(papers: List[Dict]) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for paper in papers:
        for text in splitter.split_text(paper["full_text"]):
            chunk = {"text": text}
            chunk.update({k: paper.get(k, "") for k in METADATA_KEYS})
            chunks.append(chunk)
    return chunks


def build_faiss(papers: List[Dict], embeddings) -> FAISS:
    chunks = chunk_papers(papers)
    return FAISS.from_texts(
        texts=[c["text"] for c in chunks],
        embedding=embeddings,
        metadatas=[{k: c[k] for k in METADATA_KEYS} for c in chunks],
    )
