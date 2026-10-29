import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from conftest import FakeSearch, fake_components
from medpub.generation import generate_answer, unique_sources
from medpub.pipeline import AskParams, answer_question
from medpub.queries import run_query, unique_union


class RecordingRetriever:
    def __init__(self):
        self.queries = []

    def invoke(self, q):
        self.queries.append(q)
        return [Document(page_content=f"doc for {q}", metadata={"pmid": str(len(self.queries))})]


def test_unknown_query_method_raises():
    with pytest.raises(ValueError):
        run_query("q", "magic", RecordingRetriever(), FakeListChatModel(responses=["x"]))


def test_hyde_searches_with_generated_passage():
    r = RecordingRetriever()
    run_query("does metformin help?", "hyde", r, FakeListChatModel(responses=["a hypothetical passage"]))
    assert r.queries == ["a hypothetical passage"]


def test_multiquery_includes_original_and_variants():
    r = RecordingRetriever()
    run_query("q0", "multiquery", r, FakeListChatModel(responses=["q1\nq2\n\nq3"]))
    assert r.queries == ["q0", "q1", "q2", "q3"]


def test_unique_union_drops_duplicates():
    d = Document(page_content="same", metadata={"pmid": "1"})
    assert len(unique_union([[d], [d]])) == 1


def test_sources_are_unique_per_pmid():
    docs = [Document(page_content="a", metadata={"pmid": "1", "title": "T"}),
            Document(page_content="b", metadata={"pmid": "1", "title": "T"}),
            Document(page_content="c", metadata={"pmid": "2", "title": "U"})]
    assert [s["pmid"] for s in unique_sources(docs)] == ["1", "2"]


def test_no_documents_gives_insufficient_evidence_without_calling_llm():
    answer, sources = generate_answer("q", [], llm=None)
    assert answer.startswith("Insufficient evidence") and sources == []


def test_relevant_documents_skip_the_corrective_search(settings):
    search = FakeSearch()
    out = answer_question(AskParams(question="metformin heart"), settings,
                          fake_components(grade=lambda q, d: True, search=search))
    assert out["nodes"] == ["retrieve", "grade_documents", "generate"]
    assert len(search.calls) == 1
    assert out["answer"].startswith("Answer citing") and out["sources"]


def test_irrelevant_documents_trigger_rewrite_and_research(settings):
    search = FakeSearch()
    out = answer_question(AskParams(question="metformin heart"), settings,
                          fake_components(grade=lambda q, d: "Metformin" in d, search=search))
    assert out["nodes"] == ["retrieve", "grade_documents", "transform_query", "research_again", "generate"]
    assert search.calls == ["metformin heart", "rewritten query about metformin"]


def test_no_papers_returns_insufficient_evidence(settings):
    comps = fake_components(search=lambda **kw: [])
    out = answer_question(AskParams(question="unknown topic"), settings, comps)
    assert out["n_papers"] == 0 and out["answer"].startswith("Insufficient evidence")
