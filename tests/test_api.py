import json

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from app import auth, tools
from app.agents.graph import build_graph
from app.guardrails import clean_stream_text
from app.main import _split_ready, app


def parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in fields:
            events.append((fields["event"], json.loads(fields.get("data", "null"))))
    return events


@pytest.fixture
async def client(monkeypatch):
    async def no_mcp(refresh=False):
        return {}

    monkeypatch.setattr(tools, "mcp_tools", no_mcp)
    monkeypatch.setattr("app.main.mcp_tools", no_mcp)
    store = InMemoryStore()
    app.state.store = store
    app.state.graph = build_graph(InMemorySaver(), store)
    auth.rate_limiter.buckets.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def login(client, username, password):
    r = await client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['token']}"}


async def test_login_and_auth_errors(client):
    assert (await client.post("/auth/login", json={"username": "viewer1", "password": "nope"})).status_code == 401
    assert (await client.get("/me")).status_code == 401
    headers = await login(client, "viewer1", "viewer123")
    me = (await client.get("/me", headers=headers)).json()
    assert me["role"] == "viewer" and me["tools"] == ["knowledge_search"]


async def test_admin_endpoints_need_admin(client):
    viewer = await login(client, "viewer1", "viewer123")
    admin = await login(client, "admin1", "admin123")
    assert (await client.get("/admin/audit", headers=viewer)).status_code == 403
    r = await client.post("/admin/faults", json={"pinecone": True}, headers=admin)
    assert r.json()["pinecone"] is True
    await client.post("/admin/faults", json={"pinecone": False}, headers=admin)
    assert (await client.post("/admin/faults", json={"nonsense": True}, headers=admin)).status_code == 422


async def test_invalid_chat_request(client):
    headers = await login(client, "viewer1", "viewer123")
    r = await client.post("/chat/stream", json={"message": "", "session_id": "s1"}, headers=headers)
    assert r.status_code == 422
    r = await client.post("/chat/stream", json={"message": "hi", "session_id": "../etc"}, headers=headers)
    assert r.status_code == 422


async def test_chat_streams_activity_and_answer(client):
    headers = await login(client, "analyst1", "analyst123")
    r = await client.post("/chat/stream", json={"message": "How do we rotate TLS certificates?", "session_id": "s1"},
                          headers=headers)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done" and "answer" in kinds
    activity = [d for e, d in events if e == "activity"]
    assert {"node", "plan", "retrieval", "validation", "memory"} <= {a["type"] for a in activity}
    answer = next(d for e, d in events if e == "answer")
    assert answer["citations"] and answer["citations"][0]["title"]

    history = (await client.get("/chat/history/s1", headers=headers)).json()
    assert [m["role"] for m in history["messages"]] == ["user", "assistant"]


async def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(auth.rate_limiter, "limits", {**auth.rate_limiter.limits, "viewer": (1, 0.01)})
    headers = await login(client, "viewer1", "viewer123")
    body = {"message": "hello", "session_id": "rl"}
    assert (await client.post("/chat/stream", json=body, headers=headers)).status_code == 200
    r = await client.post("/chat/stream", json=body, headers=headers)
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0


async def test_feedback_and_health(client):
    headers = await login(client, "viewer1", "viewer123")
    r = await client.post("/feedback", json={"run_id": "6f1d2b8e-0000-4000-8000-000000000000", "score": 1},
                          headers=headers)
    assert r.json()["stored"] is True
    health = (await client.get("/health")).json()
    assert health["dependencies"]["mcp"] == "unavailable"
    assert set(health["circuit_breakers"]) == {"pinecone", "mcp"}


def test_stream_filters_see_whole_values_across_chunks():
    chunks = ["The card ", "4111 11", "11 1111 ", "1111 failed. See ", "![x](https://evil.exa",
              "mple/a) and [RB-0", "01#2] done."]
    buffer, sent = "", ""
    for chunk in chunks:
        ready, buffer = _split_ready(buffer + chunk)
        sent += clean_stream_text(ready)
    sent += clean_stream_text(buffer)
    assert sent == "The card [REDACTED CARD] failed. See  and [RB-001#2] done."
