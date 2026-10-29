"""Runtime settings, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

ALLOWED_MODELS = ("gpt-4o-mini", "gpt-4o")
QUERY_METHODS = ("original", "multiquery", "hyde", "stepback")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    entrez_email: str = ""          # NCBI asks every Entrez client to identify itself
    access_code: str = ""           # empty = no access code required
    daily_limit: int = 30           # questions per client IP per day; 0 = unlimited
    grader_model: str = "gpt-4o-mini"
    allowed_models: tuple = field(default=ALLOWED_MODELS)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        entrez_email=os.getenv("ENTREZ_EMAIL", ""),
        access_code=os.getenv("APP_ACCESS_CODE", ""),
        daily_limit=int(os.getenv("DAILY_LIMIT", "30")),
        grader_model=os.getenv("GRADER_MODEL", "gpt-4o-mini"),
    )
