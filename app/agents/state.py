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
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str

    session_id: str
    question: str
    input_check: dict
    memory: dict
    plan: dict
    steps: list[str]
    evidence: Annotated[list[dict], merge_evidence]
    retrieval: dict
    research: dict
    tool_results: Annotated[list[dict], operator.add]
    pending_action: dict | None
    draft: str
    draft_source: str
    attempts: int
    validation: dict
    answer: str
    citations: list[str]
    errors: Annotated[list[dict], operator.add]
    notes: Annotated[list[str], operator.add]
