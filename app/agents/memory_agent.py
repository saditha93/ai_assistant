"""Memory loader and saver nodes. The design is described in app/memory.py."""

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from app import memory
from app.agents.response_agent import as_messages
from app.auth import User
from app.llm import llm, llm_available

SUMMARY_PROMPT = """Update the running summary of a conversation between a bank employee and the internal
assistant. Keep facts, decisions, document ids and open questions. At most 120 words.

Current summary: {summary}

New turns to fold in:
{turns}"""


async def load_memory(state: dict, runtime: Runtime[User]) -> dict:
    user, store = runtime.context, runtime.store
    context = await memory.load_long_term(store, user, state["question"]) if store else {}
    runtime.stream_writer({
        "type": "memory", "status": "loaded",
        "session_turns": len(state.get("messages", [])) // 2,
        "has_summary": bool(state.get("summary")),
        "facts": context.get("facts", []),
        "related_questions": [r["question"] for r in context.get("related", [])],
    })
    return {"memory": context}


async def _summarise(summary: str, older: list) -> str:
    turns = "\n".join(f"{m.type}: {m.text[:600]}" for m in older)
    if llm_available():
        try:
            reply = await llm(name="memory_summary").ainvoke(
                [HumanMessage(SUMMARY_PROMPT.format(summary=summary or "(empty)", turns=turns))])
            return reply.text.strip()
        except Exception:  # a failed summary must not lose the turn; fall through to truncation
            pass
    return f"{summary}\n{turns}"[-1500:]


async def save_memory(state: dict, runtime: Runtime[User]) -> dict:
    user, store = runtime.context, runtime.store
    new_messages = as_messages(state["question"], state["answer"])
    older, _ = memory.split_history(state.get("messages", []) + new_messages)
    update: dict = {"messages": new_messages}
    if older:
        update["summary"] = await _summarise(state.get("summary", ""), older)
        update["messages"] = new_messages + memory.removals(older)

    facts = []
    if store:
        facts = await memory.save_long_term(store, user, state.get("session_id", ""), state["question"],
                                            state["answer"], state.get("citations", []))
    runtime.stream_writer({"type": "memory", "status": "saved", "summarised_messages": len(older),
                           "new_facts": facts, "stored_interaction": bool(store)})
    return update
