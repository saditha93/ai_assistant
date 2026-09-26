import datetime as dt
import math
import re
import time
from zoneinfo import ZoneInfo

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings
from app.resilience import FAULTS


class LLMUnavailable(Exception):
    pass


# Free tier: every model has its own daily and per-minute quota. We remember which models
# answered 429 and skip them until their quota is back, instead of waiting on failing calls.
EXHAUSTED: dict[str, float] = {}  # model -> time.time() when it can be tried again
_limiters: dict[str, InMemoryRateLimiter] = {}


def _next_pacific_midnight() -> float:
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    return (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0).timestamp()


def record_quota_error(model: str, error: str) -> None:
    if "RESOURCE_EXHAUSTED" not in error and "429" not in error:
        return
    if "PerDay" in error:
        EXHAUSTED[model] = _next_pacific_midnight()
    else:
        delay = re.search(r"retry in ([\d.]+)s", error)
        EXHAUSTED[model] = time.time() + (float(delay.group(1)) if delay else 60)


def available_models(names: list[str]) -> list[str]:
    now = time.time()
    return [m for m in names if EXHAUSTED.get(m, 0) <= now]


class QuotaWatcher(BaseCallbackHandler):
    def __init__(self, model: str):
        self.model = model

    def on_llm_error(self, error: BaseException, **kwargs) -> None:
        record_quota_error(self.model, str(error))


def _split(names: str) -> list[str]:
    return [n.strip() for n in names.split(",") if n.strip()]


def _chat(model: str) -> ChatGoogleGenerativeAI:
    if model not in _limiters:
        _limiters[model] = InMemoryRateLimiter(requests_per_second=settings.gemini_rpm / 60,
                                               check_every_n_seconds=0.2, max_bucket_size=2)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=1,
        thinking_level=settings.gemini_thinking_level,
        rate_limiter=_limiters[model],
        callbacks=[QuotaWatcher(model)],
    )


def model_chain(light: bool = False) -> list[str]:
    main, lite = _split(settings.gemini_model), _split(settings.gemini_fallback_model)
    return available_models(lite + main if light else main + lite)


def llm_available() -> bool:
    return settings.has_llm and not FAULTS["llm_all"] and bool(model_chain())


def llm(*, name: str, tools: list | None = None, schema: type | None = None, light: bool = False) -> Runnable:
    """Try the configured models in order, skipping any whose free-tier quota is used up.
    light=True puts the lite models first for cheap, high-volume calls."""
    if not llm_available():
        raise LLMUnavailable("No LLM available (missing key, quota used up, or fault injected)")

    def build(model_name: str) -> Runnable:
        model: Runnable = _chat(model_name)
        if tools:
            model = model.bind_tools(tools)
        if schema:
            model = model.with_structured_output(schema)
        return model

    chain = model_chain(light)
    if FAULTS["llm_primary"]:
        chain = ["gemini-fault-injected", *chain]
    first, *rest = [build(m) for m in chain]
    return (first.with_fallbacks(rest) if rest else first).with_config(run_name=name)


def _embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embed_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embed_dim,
    )


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def embed_query(text: str) -> list[float]:
    if not settings.has_llm or FAULTS["llm_all"]:
        raise LLMUnavailable("Embeddings unavailable")
    return _normalise(await _embedder().aembed_query(text, task_type="RETRIEVAL_QUERY"))


async def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = await _embedder().aembed_documents(texts, task_type="RETRIEVAL_DOCUMENT")
    assert len(vectors) == len(texts), "embedding API returned a different number of vectors"
    return [_normalise(v) for v in vectors]
