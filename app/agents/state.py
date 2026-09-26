"""The shared state every agent reads and writes.

How agents share work:
  * Each agent owns a slice of the state (the supervisor writes `plan`, the retrieval
    agent adds to `evidence`, the tool agent adds to `tool_results` ...). Nobody edits
    another agent's slice, so the order they run in cannot corrupt data.
  * List fields have reducers, so two agents can add evidence without overwriting each other.
  * Failures are data: a crashing agent adds an entry to `errors` instead of stopping the
    graph. The next agents see it and degrade (e.g. the responder says a tool was down).
    This is how we stop one failure from cascading through the whole run.

Who the user is does not live here. It is passed as LangGraph runtime context (see
app.auth.User), which nodes can read but never modify, and which the LLM never sees as
editable data.

Per-turn fields are reset at the start of every turn by the input guard; `messages`,
`summary` and the checkpointed history persist across turns.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


def merge_evidence(current: list[dict], new: list[dict]) -> list[dict]:
    """Add chunks we have not seen yet, so two agents finding the same passage cite it once."""
    merged = list(current)
    seen = {c["id"] for c in current}
    for chunk in new:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            merged.append(chunk)
    return merged


class AgentState(TypedDict, total=False):
    # persistent conversation memory
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str

    # this turn
    session_id: str
    question: str
    input_check: dict
    memory: dict
    plan: dict
    steps: list[str]  # remaining worker steps chosen by the supervisor
    evidence: Annotated[list[dict], merge_evidence]
    retrieval: dict
    research: dict
    tool_results: Annotated[list[dict], operator.add]
    pending_action: dict | None
    draft: str
    draft_source: str  # llm | extractive | canned
    attempts: int
    validation: dict
    answer: str
    citations: list[str]
    errors: Annotated[list[dict], operator.add]
    notes: Annotated[list[str], operator.add]  # things the final answer must mention
