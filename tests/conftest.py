"""Shared fakes: no network, no OpenAI key needed."""
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from medpub.config import Settings  # noqa: E402
from medpub.pipeline import Components  # noqa: E402
from medpub.pubmed import full_text  # noqa: E402


def make_paper(pmid, title, abstract, year="2023"):
    paper = {"pmid": pmid, "title": title, "abstract": abstract, "authors": "Doe J",
             "journal": "J Test", "year": year, "doi": "", "types": []}
    paper["full_text"] = full_text(paper)
    return paper


PAPERS = [
    make_paper("111", "Metformin and cardiovascular outcomes", "Metformin reduced cardiovascular events."),
    make_paper("222", "Statins in older adults", "Statins lowered LDL cholesterol in adults over 75."),
    make_paper("333", "Exercise and blood pressure", "Aerobic exercise lowered systolic blood pressure."),
]


class FakeSearch:
    """Records every query; returns PAPERS for the first call, `second` afterwards."""

    def __init__(self, second=None):
        self.calls = []
        self.second = PAPERS[:1] if second is None else second

    def __call__(self, query, email, max_results, start_year, end_year):
        self.calls.append(query)
        return PAPERS if len(self.calls) == 1 else self.second


def fake_components(grade=lambda q, d: True, search=None, answer="Answer citing PMID 111 (Doe, 2023)."):
    llm = FakeListChatModel(responses=["rewritten query about metformin"])
    return Components(
        search=search or FakeSearch(),
        embeddings=DeterministicFakeEmbedding(size=32),
        query_llm=llm,
        answer_llm=FakeListChatModel(responses=[answer]),
        rewrite_llm=llm,
        grade=grade,
    )


@pytest.fixture
def settings():
    return Settings(openai_api_key="test-key", entrez_email="test@example.com", daily_limit=3)
