import os

import pytest

from app.config import settings

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Tests never call real services, even when .env has keys."""
    monkeypatch.setattr(settings, "google_api_key", "")
    monkeypatch.setattr(settings, "pinecone_api_key", "")
    monkeypatch.setattr(settings, "langsmith_api_key", "")
