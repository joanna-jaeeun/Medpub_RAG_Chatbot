"""FastAPI service for the MedPub RAG chatbot.

Run locally:  uvicorn api.main:app --reload
Docs:         http://localhost:8000/docs
"""
from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from threading import Lock
from typing import Callable, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from medpub import __version__
from medpub.config import Settings, get_settings
from medpub.pipeline import AskParams, answer_question
from medpub.pubmed import PubMedError

log = logging.getLogger("medpub.api")
app = FastAPI(title="MedPub RAG API", version=__version__,
              description="PubMed-grounded medical literature Q&A (CRAG). Not medical advice.")

THIS_YEAR = dt.date.today().year


# ---------------------------------------------------------------- schemas

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    query_method: Literal["original", "multiquery", "hyde", "stepback"] = "original"
    max_results: int = Field(10, ge=5, le=50)
    start_year: int = Field(2020, ge=1950, le=THIS_YEAR)
    end_year: int = Field(THIS_YEAR, ge=1950, le=THIS_YEAR)
    model: Literal["gpt-4o-mini", "gpt-4o"] = "gpt-4o-mini"

    @model_validator(mode="after")
    def years_in_order(self):
        if self.start_year > self.end_year:
            raise ValueError("start_year must not be after end_year")
        return self


class Source(BaseModel):
    pmid: str
    title: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    abstract: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    nodes: List[str] = Field(description="CRAG nodes that ran, in order")
    n_papers: int
    latency_s: float


# ---------------------------------------------------------------- cost guard

class DailyLimiter:
    """In-memory per-client daily counter. Resets when the process restarts,
    which is acceptable for a single small instance."""

    def __init__(self):
        self._counts = defaultdict(int)
        self._lock = Lock()

    def hit(self, client: str, limit: int) -> bool:
        if limit <= 0:
            return True
        key = (client, dt.date.today().isoformat())
        with self._lock:
            if self._counts[key] >= limit:
                return False
            self._counts[key] += 1
            return True


limiter = DailyLimiter()


def client_id(request: Request) -> str:
    # Render and most proxies put the real client first in X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def guard(request: Request, settings: Settings = Depends(get_settings),
          x_access_code: Optional[str] = Header(default=None)) -> Settings:
    if settings.access_code and x_access_code != settings.access_code:
        raise HTTPException(status_code=401, detail="Missing or wrong access code")
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="Server has no OpenAI API key configured")
    if not limiter.hit(client_id(request), settings.daily_limit):
        raise HTTPException(status_code=429, detail=f"Daily limit of {settings.daily_limit} questions reached")
    return settings


def get_answerer() -> Callable[[AskParams, Settings], dict]:
    return answer_question


# ---------------------------------------------------------------- routes

@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {"status": "ok", "version": __version__,
            "openai_key_set": bool(settings.openai_api_key),
            "access_code_required": bool(settings.access_code)}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, settings: Settings = Depends(guard),
        answerer: Callable = Depends(get_answerer)):
    # A plain (sync) endpoint: FastAPI runs it in a worker thread, so slow
    # PubMed/OpenAI calls do not block the server.
    params = AskParams(**body.model_dump())
    try:
        result = answerer(params, settings)
    except PubMedError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception:
        log.exception("ask failed")
        raise HTTPException(status_code=500, detail="Internal error while answering")
    log.info("ask method=%s papers=%s latency=%.2fs nodes=%s", body.query_method,
             result["n_papers"], result["latency_s"], ",".join(result["nodes"]))
    return result
