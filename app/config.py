import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent

# LangSmith and LangGraph read their settings (LANGSMITH_*, LANGGRAPH_STRICT_MSGPACK) from
# the real environment, so .env has to be loaded into os.environ, not only into Settings.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    brand_name: str = "Crestline Commercial Bank"
    assistant_name: str = "Crest"

    # Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    llm_timeout_s: float = 45

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index: str = "crestline-kb"
    pinecone_region: str = "us-east-1"
    rerank_model: str = "bge-reranker-v2-m3"
    hybrid_alpha: float = 0.6  # weight of the dense score; sparse gets 1 - alpha
    retrieval_top_k: int = 6
    min_rerank_score: float = 0.15  # below this the retrieval agent rewrites the query once

    # MCP
    mcp_url: str = "http://localhost:8001/mcp"
    tool_timeout_s: float = 10

    # LangSmith reads LANGSMITH_* itself; we only need to know whether it is on.
    langsmith_api_key: str = ""
    langsmith_project: str = "crestline-assistant"

    # Auth and rate limiting
    jwt_secret: str = "dev-only-secret-change-me-before-deploying"
    jwt_ttl_minutes: int = 480
    # role -> (bucket capacity, tokens refilled per second)
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
    os.environ["LANGSMITH_TRACING"] = "false"  # avoid "missing API key" warnings on every run
