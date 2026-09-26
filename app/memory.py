import datetime as dt
import re
import uuid

from langchain_core.messages import AnyMessage, RemoveMessage
from langgraph.store.base import BaseStore

from app.auth import User
from app.retrieval.bm25 import BM25

KEEP_TURNS = 6
HISTORY_SCAN = 50

FACT_PATTERNS = [
    re.compile(r"\b(i work (on|in|for|with)|i'm (on|in) the|i am (on|in) the)\b", re.IGNORECASE),
    re.compile(r"\b(i prefer|please always|always give me)\b", re.IGNORECASE),
    re.compile(r"\b(call me|my name is)\b", re.IGNORECASE),
]


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def extract_facts(message: str) -> list[str]:
    sentences = (s.strip() for s in re.split(r"(?<=[.!?])\s+", message))
    return [s.rstrip(".") for s in sentences if any(p.search(s) for p in FACT_PATTERNS)]


async def load_long_term(store: BaseStore, user: User, question: str) -> dict:
    facts = [item.value["fact"] for item in await store.asearch((user.username, "facts"), limit=20)]
    items = await store.asearch((user.username, "interactions"), limit=HISTORY_SCAN)
    past = sorted((i.value for i in items), key=lambda v: v["ts"], reverse=True)
    related = []
    if past:
        ranking = BM25([p["question"] for p in past]).search(question, k=3)
        related = [past[i] for i, score in ranking if score > 0.2]
    return {
        "profile": {"name": user.name, "role": user.role, "department": user.department},
        "facts": facts,
        "related": related,
    }


async def save_long_term(store: BaseStore, user: User, session_id: str, question: str, answer: str,
                         citations: list[str]) -> list[str]:
    await store.aput((user.username, "interactions"), str(uuid.uuid4()), {
        "ts": _now(), "session_id": session_id, "question": question,
        "answer_summary": answer[:300], "citations": citations,
    })
    new_facts = extract_facts(question)
    for fact in new_facts:
        await store.aput((user.username, "facts"), fact.lower()[:60], {"fact": fact, "ts": _now()})
    return new_facts


def split_history(messages: list[AnyMessage]) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """Return (older messages to summarise, recent messages to keep verbatim)."""
    keep = KEEP_TURNS * 2
    if len(messages) <= keep:
        return [], messages
    return messages[:-keep], messages[-keep:]


def removals(messages: list[AnyMessage]) -> list[RemoveMessage]:
    return [RemoveMessage(id=m.id) for m in messages]


async def audit(store: BaseStore | None, user: User, event: str, detail: dict) -> None:
    if store is None:
        return
    await store.aput(("audit",), str(uuid.uuid4()), {
        "ts": _now(), "user": user.username, "role": user.role, "event": event, "detail": detail,
    })


async def read_audit(store: BaseStore, limit: int = 50) -> list[dict]:
    items = await store.asearch(("audit",), limit=500)
    return sorted((i.value for i in items), key=lambda v: v["ts"], reverse=True)[:limit]


async def save_feedback(store: BaseStore, user: User, run_id: str, score: int, comment: str | None) -> None:
    await store.aput(("feedback",), run_id, {
        "ts": _now(), "user": user.username, "run_id": run_id, "score": score, "comment": comment,
    })


def format_for_prompt(context: dict, summary: str) -> str:
    """Memory as a short text block for prompts."""
    if not context:
        return "(none)"
    lines = []
    profile = context.get("profile")
    if profile:
        lines.append(f"User: {profile['name']}, {profile['role']}, {profile['department']} department.")
    if context.get("facts"):
        lines.append("Known about the user: " + "; ".join(context["facts"]))
    if summary:
        lines.append(f"Earlier in this conversation: {summary}")
    for past in context.get("related", []):
        lines.append(f"Previously asked ({past['ts'][:10]}): {past['question']} -> {past['answer_summary'][:150]}")
    return "\n".join(lines) or "(none)"
