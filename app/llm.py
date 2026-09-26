"""Gemini chat models and embeddings.

Model choice (see docs/architecture.md for the longer version):
  * gemini-3.8-flash for every agent: fast, cheap, 1M-token context, reliable tool calling
    and JSON-schema structured output. A Pro model would be slower and costlier for little
    gain on retrieval-grounded answers.
  * gemini-3.5-flash-lite as the automatic fallback when the primary call fails.
  * gemini-embedding-001 at 768 dimensions: good retrieval quality at a quarter of the
    storage of the full 3072 dimensions.

Every LLM call goes through llm(), so fallback, timeouts and fault injection behave the
same everywhere.
"""

import math

from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import settings
from app.resilience import FAULTS


class LLMUnavailable(Exception):
    pass


def _chat(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        timeout=settings.llm_timeout_s,
        max_retries=2,
        thinking_level=settings.gemini_thinking_level,
    )


def llm_available() -> bool:
    return settings.has_llm and not FAULTS["llm_all"]


def llm(*, name: str, tools: list | None = None, schema: type | None = None, light: bool = False) -> Runnable:
    """Primary model with the lite model as fallback, optionally with tools or a
    structured-output schema applied to both.

    light=True flips the order for cheap, high-volume calls (research sub-queries, query
    rewrites, summaries): the lite model answers first and the main model is the fallback.
    That keeps most of the main model's quota for planning and final answers."""
    if not llm_available():
        raise LLMUnavailable("No LLM available (missing GOOGLE_API_KEY or fault injected)")

    def build(model_name: str) -> Runnable:
        model: Runnable = _chat(model_name)
        if tools:
            model = model.bind_tools(tools)
        if schema:
            model = model.with_structured_output(schema)
        return model

    # The fault switch swaps in a model name that does not exist, so the failure and the
    # fallback are real API behaviour rather than a mock.
    first, second = settings.gemini_model, settings.gemini_fallback_model
    if light:
        first, second = second, first
    if FAULTS["llm_primary"]:
        first = "gemini-fault-injected"
    return build(first).with_fallbacks([build(second)]).with_config(run_name=name)


def _embedder() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embed_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embed_dim,
    )


def _normalise(vec: list[float]) -> list[float]:
    # Below 3072 dimensions Gemini vectors are not unit length. We normalise so the
    # dot product in Pinecone equals cosine similarity.
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def embed_query(text: str) -> list[float]:
    if not llm_available():
        raise LLMUnavailable("Embeddings unavailable")
    return _normalise(await _embedder().aembed_query(text, task_type="RETRIEVAL_QUERY"))


async def embed_documents(texts: list[str]) -> list[list[float]]:
    vectors = await _embedder().aembed_documents(texts, task_type="RETRIEVAL_DOCUMENT")
    assert len(vectors) == len(texts), "embedding API returned a different number of vectors"
    return [_normalise(v) for v in vectors]
