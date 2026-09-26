import asyncio
import time
from collections.abc import Awaitable, Callable

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from app.agents.guard import guard_input, refuse
from app.agents.memory_agent import load_memory, save_memory
from app.agents.research_agent import research_agent
from app.agents.response_agent import MAX_ATTEMPTS, extractive_answer, responder, validator
from app.agents.retrieval_agent import retrieval_agent
from app.agents.state import AgentState
from app.agents.supervisor import keyword_plan, supervisor
from app.agents.tool_agent import approve_action, tool_agent
from app.auth import User
from app.guardrails import REFUSAL
from app.observability import log

NodeFn = Callable[[dict, Runtime[User]], Awaitable[dict]]


def describe_error(exc: Exception) -> str:
    """A short message for state, the UI and the answer. Full details go to the log."""
    text = str(exc)
    if isinstance(exc, TimeoutError):
        return "timed out"
    if "RESOURCE_EXHAUSTED" in text or "429" in text:
        return "language model quota or rate limit reached"
    return f"{type(exc).__name__}: {text[:160]}"
WORKER_FOR_STEP = {"retrieve": "retrieval_agent", "research": "research_agent", "tools": "tool_agent"}


def node(name: str, fn: NodeFn, *, timeout: float = 60, pops_step: bool = False,
         on_error: Callable[[dict, Exception], dict] | None = None) -> NodeFn:
    async def run(state: AgentState, runtime: Runtime[User]) -> dict:
        emit = runtime.stream_writer
        emit({"type": "node", "node": name, "status": "start"})
        started = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                update = await fn(state, runtime) or {}
            status = "end"
        except GraphBubbleUp:
            raise
        except Exception as exc:
            log.exception("node_failed", node=name)
            error = describe_error(exc)
            update = on_error(state, exc) if on_error else {}
            update["errors"] = update.get("errors", []) + [{"node": name, "error": error}]
            emit({"type": "node", "node": name, "status": "error", "error": error})
            status = "failed"
        if pops_step:
            update["steps"] = state.get("steps", [])[1:]
        emit({"type": "node", "node": name, "status": status,
              "ms": round((time.perf_counter() - started) * 1000)})
        return update

    run.__name__ = name
    return run


def next_step(state: AgentState) -> str:
    steps = state.get("steps") or []
    return WORKER_FOR_STEP.get(steps[0], "responder") if steps else "responder"


def after_tools(state: AgentState) -> str:
    return "approve_action" if state.get("pending_action") else next_step(state)


def after_validation(state: AgentState) -> str:
    return "save_memory" if state.get("answer") or state.get("attempts", 0) >= MAX_ATTEMPTS else "responder"


def build_graph(checkpointer=None, store=None):
    g = StateGraph(AgentState, context_schema=User)

    g.add_node("guard_input", node("guard_input", guard_input, timeout=5,
                                   on_error=lambda s, e: {"input_check": {"allowed": False, "reason": "guard error"}}))
    g.add_node("refuse", node("refuse", refuse, timeout=5, on_error=lambda s, e: {"answer": REFUSAL}))
    g.add_node("load_memory", node("load_memory", load_memory, timeout=10))
    g.add_node("supervisor", node("supervisor", supervisor, timeout=45, on_error=lambda s, e: {
        "plan": {**keyword_plan(s["question"]), "steps": ["retrieve"]}, "steps": ["retrieve"]}))
    g.add_node("retrieval_agent", node("retrieval_agent", retrieval_agent, timeout=45, pops_step=True))
    g.add_node("research_agent", node("research_agent", research_agent, timeout=240, pops_step=True))
    g.add_node("tool_agent", node("tool_agent", tool_agent, timeout=90, pops_step=True))
    g.add_node("approve_action", node("approve_action", approve_action, timeout=30,
                                      on_error=lambda s, e: {"pending_action": None}))
    g.add_node("responder", node("responder", responder, timeout=90, on_error=lambda s, e: {
        "draft": extractive_answer(s, "answer generation failed"), "draft_source": "extractive"}))
    g.add_node("validator", node("validator", validator, timeout=10, on_error=lambda s, e: {
        "answer": "I'm sorry, something went wrong while checking my answer. Please try again."}))
    g.add_node("save_memory", node("save_memory", save_memory, timeout=30))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", lambda s: "load_memory" if s["input_check"]["allowed"] else "refuse",
                            ["load_memory", "refuse"])
    g.add_edge("refuse", END)
    g.add_edge("load_memory", "supervisor")
    routes = [*WORKER_FOR_STEP.values(), "responder"]
    for source in ("supervisor", "retrieval_agent", "research_agent", "approve_action"):
        g.add_conditional_edges(source, next_step, routes)
    g.add_conditional_edges("tool_agent", after_tools, [*routes, "approve_action"])
    g.add_edge("responder", "validator")
    g.add_conditional_edges("validator", after_validation, ["responder", "save_memory"])
    g.add_edge("save_memory", END)
    return g.compile(checkpointer=checkpointer, store=store)
