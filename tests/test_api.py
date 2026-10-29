import pytest
from fastapi.testclient import TestClient

from api import main
from medpub.config import Settings
from medpub.pubmed import PubMedError

FAKE_RESULT = {"answer": "ok", "sources": [{"pmid": "111", "title": "T"}],
               "nodes": ["retrieve", "grade_documents", "generate"], "n_papers": 3, "latency_s": 0.1}


@pytest.fixture
def client():
    main.limiter = main.DailyLimiter()          # fresh counter per test
    main.app.dependency_overrides[main.get_answerer] = lambda: (lambda params, settings: FAKE_RESULT)
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def use_settings(**kw):
    base = dict(openai_api_key="k", entrez_email="e@x.com", daily_limit=3)
    base.update(kw)
    main.app.dependency_overrides[main.get_settings] = lambda: Settings(**base)


def test_health(client):
    use_settings()
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["openai_key_set"] is True


def test_ask_returns_answer_and_sources(client):
    use_settings()
    resp = client.post("/ask", json={"question": "Does metformin help?"})
    assert resp.status_code == 200
    assert resp.json()["sources"][0]["pmid"] == "111"


@pytest.mark.parametrize("payload", [
    {"question": "hi"},                                               # too short
    {"question": "valid question", "query_method": "magic"},          # unknown method
    {"question": "valid question", "start_year": 2024, "end_year": 2020},
    {"question": "valid question", "max_results": 500},
])
def test_invalid_requests_are_rejected(client, payload):
    use_settings()
    assert client.post("/ask", json=payload).status_code == 422


def test_access_code_required_when_configured(client):
    use_settings(access_code="secret")
    assert client.post("/ask", json={"question": "a question"}).status_code == 401
    ok = client.post("/ask", json={"question": "a question"}, headers={"X-Access-Code": "secret"})
    assert ok.status_code == 200


def test_daily_limit(client):
    use_settings(daily_limit=2)
    codes = [client.post("/ask", json={"question": "a question"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_missing_openai_key_gives_503(client):
    use_settings(openai_api_key="")
    assert client.post("/ask", json={"question": "a question"}).status_code == 503


def test_pubmed_failure_gives_502(client):
    use_settings()

    def failing(params, settings):
        raise PubMedError("down")

    main.app.dependency_overrides[main.get_answerer] = lambda: failing
    assert client.post("/ask", json={"question": "a question"}).status_code == 502
