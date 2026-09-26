import asyncio

import pytest

from app import resilience
from app.resilience import CircuitBreaker, DependencyDown


async def test_breaker_opens_after_failures_and_recovers(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(resilience.time, "monotonic", lambda: clock[0])
    breaker = CircuitBreaker("dep", failure_threshold=2, reset_after_s=10)

    async def boom():
        raise ConnectionError("down")

    async def ok():
        return "ok"

    for _ in range(2):
        with pytest.raises(ConnectionError):
            await breaker.call(boom, timeout=1)
    assert breaker.state == "open"
    with pytest.raises(DependencyDown):  # fails fast, dependency not called
        await breaker.call(ok, timeout=1)

    clock[0] += 10
    assert breaker.state == "half_open"
    assert await breaker.call(ok, timeout=1) == "ok"
    assert breaker.state == "closed"


async def test_timeout_counts_as_failure():
    breaker = CircuitBreaker("slow", failure_threshold=1)

    async def hang():
        await asyncio.sleep(5)

    with pytest.raises(TimeoutError):
        await breaker.call(hang, timeout=0.05)
    assert breaker.state == "open"


async def test_fault_switch(monkeypatch):
    monkeypatch.setitem(resilience.FAULTS, "pinecone", True)

    async def ok():
        return "ok"

    with pytest.raises(DependencyDown):
        await CircuitBreaker("pinecone").call(ok, timeout=1)
