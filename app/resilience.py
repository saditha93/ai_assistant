"""Circuit breakers for external dependencies, plus fault switches for the demo.

Retries already happen inside the clients (Gemini max_retries, Pinecone's built-in retry),
so this module only adds what they don't: a timeout per call and a breaker that stops us
from hammering a dependency that is clearly down. While a breaker is open, callers get a
DependencyDown error immediately and fall back to their degraded path.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

# Admins can flip these from the UI to show graceful degradation live.
FAULTS = {
    "llm_primary": False,  # primary Gemini model fails -> fallback model answers
    "llm_all": False,  # every LLM call fails -> extractive answer from retrieved passages
    "pinecone": False,  # vector DB down -> local keyword search
    "mcp": False,  # MCP server down -> tools reported unavailable
    "tool_timeout": False,  # tools hang -> per-call timeout fires
}


class DependencyDown(Exception):
    pass


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, reset_after_s: float = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_after_s = reset_after_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        if time.monotonic() - self.opened_at >= self.reset_after_s:
            return "half_open"  # let one call through to test the dependency
        return "open"

    async def call(self, fn: Callable[..., Awaitable[Any]], *args: Any, timeout: float, **kwargs: Any) -> Any:
        if FAULTS.get(self.name):
            self._record_failure()
            raise DependencyDown(f"{self.name} unavailable (fault injected)")
        if self.state == "open":
            raise DependencyDown(f"{self.name} unavailable (circuit open)")
        try:
            async with asyncio.timeout(timeout):
                result = await fn(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise
        self.failures, self.opened_at = 0, None
        return result

    def _record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


pinecone_breaker = CircuitBreaker("pinecone")
mcp_breaker = CircuitBreaker("mcp")
BREAKERS = {b.name: b for b in (pinecone_breaker, mcp_breaker)}
