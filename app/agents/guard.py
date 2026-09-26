from langgraph.runtime import Runtime
from langgraph.types import Overwrite

from app import guardrails, memory
from app.agents.response_agent import as_messages
from app.auth import User

TURN_RESET = {
    "evidence": Overwrite([]), "tool_results": Overwrite([]), "errors": Overwrite([]), "notes": Overwrite([]),
    "plan": {}, "steps": [], "memory": {}, "retrieval": {}, "research": {}, "pending_action": None,
    "draft": "", "draft_source": "", "attempts": 0, "validation": {}, "answer": "", "citations": [],
}


async def guard_input(state: dict, runtime: Runtime[User]) -> dict:
    check = guardrails.check_user_input(state["question"])
    runtime.stream_writer({"type": "guard", "allowed": check["allowed"], "flagged": check.get("flagged", False),
                           "score": check["score"], "labels": check["labels"], "reason": check.get("reason")})
    if not check["allowed"] or check.get("flagged"):
        await memory.audit(runtime.store, runtime.context,
                           "input_blocked" if not check["allowed"] else "input_flagged",
                           {"labels": check["labels"], "score": check["score"], "text": check["text"][:200]})
    return {**TURN_RESET, "input_check": check, "question": check["text"] or state["question"]}


async def refuse(state: dict, runtime: Runtime[User]) -> dict:
    reason = state["input_check"].get("reason")
    answer = guardrails.REFUSAL
    if reason == "message too long":
        answer = f"Your message is too long. Please keep it under {guardrails.MAX_MESSAGE_CHARS} characters."
    runtime.stream_writer({"type": "response", "status": "refused", "reason": reason})
    return {"answer": answer, "messages": as_messages(state["question"][:500], answer)}
