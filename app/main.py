import datetime as dt
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import memory
from app.agents.graph import build_graph
from app.auth import User, authenticate, create_token, current_user, rate_limited_user, require_admin
from app.config import settings
from app.guardrails import clean_stream_text
from app.llm import EXHAUSTED, llm_available, model_chain
from app.observability import log, send_langsmith_feedback, setup_logging
from app.resilience import BREAKERS, FAULTS
from app.tools import mcp_tools

SessionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    async with (
        AsyncSqliteSaver.from_conn_string(str(settings.state_dir / "checkpoints.sqlite")) as checkpointer,
        AsyncSqliteStore.from_conn_string(str(settings.state_dir / "memory.sqlite")) as store,
    ):
        await store.setup()
        app.state.store = store
        app.state.graph = build_graph(checkpointer, store)
        log.info("api_started", llm=settings.has_llm, pinecone=settings.has_pinecone,
                 langsmith=settings.has_langsmith)
        yield


app = FastAPI(title=f"{settings.brand_name} Assistant API", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    log.info("request", method=request.method, status=response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled_error")
    return JSONResponse({"detail": "Internal error. Please try again."}, status_code=500)

class LoginRequest(BaseModel):
    username: str = Field(max_length=50)
    password: str = Field(max_length=100)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: SessionId


class ResumeRequest(BaseModel):
    session_id: SessionId
    approved: bool


class FeedbackRequest(BaseModel):
    run_id: uuid.UUID
    score: int = Field(ge=0, le=1)
    comment: str | None = Field(None, max_length=1000)

@app.post("/auth/login")
async def login(body: LoginRequest):
    user = authenticate(body.username, body.password)
    if not user:
        raise HTTPException(401, "Invalid username or password")
    return {"token": create_token(user), "user": user.__dict__ | {"tools": sorted(user.tools)}}


@app.get("/me")
async def me(user: User = Depends(current_user)):
    return user.__dict__ | {"tools": sorted(user.tools), "access_levels": user.access_levels}

def _config(user: User, session_id: str, run_id: uuid.UUID) -> dict:
    thread_id = f"{user.username}:{session_id}"
    return {
        "configurable": {"thread_id": thread_id},
        "run_id": run_id,
        "run_name": "chat_turn",
        "recursion_limit": 40,
        "metadata": {"thread_id": thread_id, "session_id": session_id, "user": user.username, "role": user.role},
        "tags": [f"role:{user.role}"],
    }


TRAILING_NUMBER = re.compile(r"\d[\d -]*$")


def _split_ready(buffer: str) -> tuple[str, str]:
    """Decide how much streamed text is safe to send now.

    We release up to the last space, but hold back while a markdown link, image or citation
    bracket is still open, or while the text ends in digits (a card number arrives in pieces
    like "4111 1111"). That way the redaction and link filters always see whole values.
    """
    cut = max(buffer.rfind(" "), buffer.rfind("\n"))
    ready = buffer[: cut + 1]
    balanced = ready.count("[") == ready.count("]") and ready.count("(") == ready.count(")")
    if cut < 0 or not balanced or TRAILING_NUMBER.search(ready.rstrip()):
        if len(buffer) > 600:  # never stall the stream on unusual text
            return buffer, ""
        return "", buffer
    return ready, buffer[cut + 1:]


async def stream_turn(graph, payload, user: User, session_id: str) -> AsyncIterator[ServerSentEvent]:
    run_id = uuid.uuid4()
    config = _config(user, session_id, run_id)
    structlog.contextvars.bind_contextvars(user=user.username, session=session_id, run_id=str(run_id))
    yield ServerSentEvent(event="start", data={"run_id": str(run_id), "session_id": session_id})
    buffer = ""
    try:
        async for part in graph.astream(payload, config, context=user,
                                        stream_mode=["custom", "messages", "updates"], version="v2"):
            kind, data = part["type"], part["data"]
            if kind == "custom":
                if data.get("type") == "answer_reset":
                    buffer = ""
                yield ServerSentEvent(event="activity", data=data)
            elif kind == "messages":
                chunk, meta = data
                if meta.get("langgraph_node") == "responder" and chunk.text:
                    ready, buffer = _split_ready(buffer + chunk.text)
                    if ready:
                        yield ServerSentEvent(event="token", data=clean_stream_text(ready))
            elif kind == "updates" and "__interrupt__" in data:
                yield ServerSentEvent(event="approval_required", data=data["__interrupt__"][0].value)
        if buffer:
            yield ServerSentEvent(event="token", data=clean_stream_text(buffer))

        snapshot = await graph.aget_state(config)
        if snapshot.next:
            yield ServerSentEvent(event="done", data={"status": "awaiting_approval", "run_id": str(run_id)})
            return
        values = snapshot.values
        titles = {}
        for c in values.get("evidence", []):
            titles[c["id"]] = (c["title"], c["section"])
            titles.setdefault(c["doc_id"], (c["title"], "whole document"))
        validation = values.get("validation", {})
        yield ServerSentEvent(event="answer", data={
            "answer": values.get("answer", ""),
            "citations": [{"id": c, "title": titles.get(c, ("", ""))[0], "section": titles.get(c, ("", ""))[1]}
                          for c in values.get("citations", [])],
            "validation": {"passed": validation.get("passed"), "failed": validation.get("failed", []),
                           "fallback": validation.get("fallback", False)},
            "notes": values.get("notes", []),
            "errors": values.get("errors", []),
            "run_id": str(run_id),
        })
    except Exception:
        log.exception("chat_turn_failed")
        message = "The assistant hit an unexpected error. Please try again."
        yield ServerSentEvent(event="error", data={"message": message})
    yield ServerSentEvent(event="done", data={"status": "complete", "run_id": str(run_id)})


@app.post("/chat/stream", response_class=EventSourceResponse)
async def chat_stream(body: ChatRequest, request: Request, user: User = Depends(rate_limited_user)):
    payload = {"question": body.message, "session_id": body.session_id}
    async for event in stream_turn(request.app.state.graph, payload, user, body.session_id):
        yield event


@app.post("/chat/resume", response_class=EventSourceResponse)
async def chat_resume(body: ResumeRequest, request: Request, user: User = Depends(rate_limited_user)):
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": f"{user.username}:{body.session_id}"}})
    if not snapshot.next:
        yield ServerSentEvent(event="error", data={"message": "Nothing is waiting for approval in this session."})
        yield ServerSentEvent(event="done", data={"status": "complete"})
        return
    async for event in stream_turn(graph, Command(resume={"approved": body.approved}), user, body.session_id):
        yield event


@app.get("/chat/history/{session_id}")
async def history(session_id: SessionId, request: Request, user: User = Depends(current_user)):
    snapshot = await request.app.state.graph.aget_state(
        {"configurable": {"thread_id": f"{user.username}:{session_id}"}})
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    return {"summary": snapshot.values.get("summary", "") if snapshot.values else "",
            "messages": [{"role": "user" if m.type == "human" else "assistant", "content": m.text} for m in messages]}


@app.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request, user: User = Depends(current_user)):
    await memory.save_feedback(request.app.state.store, user, str(body.run_id), body.score, body.comment)
    sent = send_langsmith_feedback(str(body.run_id), body.score, body.comment)
    return {"stored": True, "sent_to_langsmith": sent}

@app.get("/health")
async def health():
    mcp_up = bool(await mcp_tools(refresh=True))
    return {
        "status": "ok",
        "dependencies": {
            "llm": "ok" if llm_available() else ("fault_injected" if settings.has_llm else "not_configured"),
            "pinecone": "configured" if settings.has_pinecone else "not_configured (keyword fallback)",
            "mcp": "ok" if mcp_up else "unavailable",
            "langsmith": "tracing" if settings.has_langsmith else "off",
        },
        "llm_models": {
            "usable": model_chain() if settings.has_llm else [],
            "quota_exhausted": {m: dt.datetime.fromtimestamp(t, dt.UTC).isoformat(timespec="minutes")
                                for m, t in EXHAUSTED.items() if t > time.time()},
        },
        "circuit_breakers": {name: b.state for name, b in BREAKERS.items()},
        "faults": FAULTS,
    }


@app.get("/admin/faults")
async def get_faults(user: User = Depends(require_admin)):
    return FAULTS


@app.post("/admin/faults")
async def set_faults(changes: dict[str, bool], request: Request, user: User = Depends(require_admin)):
    unknown = set(changes) - set(FAULTS)
    if unknown:
        raise HTTPException(422, f"Unknown faults: {sorted(unknown)}")
    FAULTS.update(changes)
    await memory.audit(request.app.state.store, user, "faults_changed", changes)
    return FAULTS


@app.get("/admin/audit")
async def audit_log(request: Request, limit: int = 50, user: User = Depends(require_admin)):
    return await memory.read_audit(request.app.state.store, min(max(limit, 1), 200))
