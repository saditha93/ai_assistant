import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    brand_name: str = "Crestline Commercial Bank"
    assistant_name: str = "Crest"

    # Gemini
    google_api_key: str = ""
    # Comma-separated. Free tier quotas are per model, so several models spread the load.
    gemini_model: str = "gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash,gemini-3.5-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    gemini_rpm: int = 10
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    gemini_thinking_level: str = "low"
    llm_timeout_s: float = 30

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "crestline-kb"
    pinecone_region: str = "us-east-1"
    rerank_model: str = "bge-reranker-v2-m3"
    hybrid_alpha: float = 0.6
    retrieval_top_k: int = 6
    min_rerank_score: float = 0.15

    # MCP
    mcp_url: str = "http://localhost:8001/mcp"
    tool_timeout_s: float = 10

    langsmith_api_key: str = ""
    langsmith_project: str = "crestline-assistant"

    jwt_secret: str = "dev-only-secret-change-me-before-deploying"
    jwt_ttl_minutes: int = 480
    rate_limits: dict[str, tuple[int, float]] = {
        "viewer": (10, 0.2),
        "analyst": (20, 0.5),
        "admin": (40, 1.0),
    }

    data_dir: Path = ROOT / "data"
    state_dir: Path = ROOT / "data" / "state"
    log_level: str = "INFO"

    @property
    def has_llm(self) -> bool:
        return bool(self.google_api_key)

    @property
    def has_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def has_langsmith(self) -> bool:
        return bool(self.langsmith_api_key)


settings = Settings()

if not settings.has_langsmith:
    os.environ["LANGSMITH_TRACING"] = "false"
