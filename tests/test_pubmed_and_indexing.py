import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding

from conftest import PAPERS
from medpub import pubmed
from medpub.indexing import CHUNK_SIZE, build_faiss, chunk_papers


def test_record_to_paper_maps_fields():
    record = {"PMID": "123", "TI": "A title", "AB": "An abstract", "AU": ["Kim A", "Lee B"],
              "TA": "Lancet", "DP": "2021 Mar 5", "AID": ["S0140 [pii]", "10.1000/xyz [doi]"],
              "PT": ["Journal Article"]}
    paper = pubmed.record_to_paper(record)
    assert paper["pmid"] == "123" and paper["year"] == "2021"
    assert paper["authors"] == "Kim A, Lee B"
    assert paper["doi"] == "10.1000/xyz"


def test_record_without_optional_fields():
    paper = pubmed.record_to_paper({"PMID": "9"})
    assert paper["title"] == "No title" and paper["year"] == "" and paper["doi"] == ""


def test_search_adds_date_filter_and_fills_missing_abstract(monkeypatch):
    seen = {}

    def fake_ids(term, n):
        seen["term"] = term
        return ["1"]

    monkeypatch.setattr(pubmed, "_esearch_ids", fake_ids)
    monkeypatch.setattr(pubmed, "_efetch_records", lambda ids: [{"PMID": "1", "TI": "T"}])
    monkeypatch.setattr(pubmed, "_efetch_abstract", lambda pmid: "  fetched abstract ")
    papers = pubmed.search_pubmed("metformin", "a@b.c", 5, 2020, 2024)
    assert seen["term"] == "metformin AND 2020:2024[dp]"
    assert papers[0]["abstract"] == "fetched abstract"
    assert "PMID: 1" in papers[0]["full_text"]


def test_search_returns_empty_list_when_no_hits(monkeypatch):
    monkeypatch.setattr(pubmed, "_esearch_ids", lambda term, n: [])
    assert pubmed.search_pubmed("nothing", "a@b.c") == []


def test_network_failure_raises_pubmed_error(monkeypatch):
    def boom(term, n):
        raise OSError("connection reset")

    monkeypatch.setattr(pubmed, "_esearch_ids", boom)
    with pytest.raises(pubmed.PubMedError):
        pubmed.search_pubmed("x", "a@b.c")


def test_chunks_keep_metadata_and_size():
    chunks = chunk_papers(PAPERS)
    assert {c["pmid"] for c in chunks} == {"111", "222", "333"}
    assert all(len(c["text"]) <= CHUNK_SIZE for c in chunks)


def test_faiss_returns_documents_with_pmid():
    docs = build_faiss(PAPERS, DeterministicFakeEmbedding(size=32)).as_retriever(
        search_kwargs={"k": 2}).invoke("metformin")
    assert len(docs) == 2 and all(d.metadata["pmid"] for d in docs)
