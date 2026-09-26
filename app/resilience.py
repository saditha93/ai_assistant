import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

FAULTS = {
    "llm_primary": False,
    "llm_all": False,
    "pinecone": False,
    "mcp": False,
    "tool_timeout": False,
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
            return "half_open"
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
