from langgraph.store.memory import InMemoryStore

from app import memory, resilience, tools
from app.auth import User
from app.tools import ToolContext, execute_tool

VIEWER = User("viewer1", "V", "viewer", "retail")
ANALYST = User("analyst1", "A", "analyst", "payments")
ADMIN = User("admin1", "R", "admin", "platform")


def ctx(user, store=None):
    events = []
    return ToolContext(user, store, events.append), events


async def test_viewer_cannot_run_analyst_tools_even_if_asked():
    store = InMemoryStore()
    c, events = ctx(VIEWER, store)
    result = await execute_tool(c, "search_incidents", {"severity": "SEV1"})
    assert not result["ok"] and "Not permitted" in result["error"]
    assert any(e["status"] == "denied" for e in events)
    audit = await memory.read_audit(store)
    assert audit[0]["event"] == "tool_denied"


async def test_tool_list_depends_on_role(monkeypatch):
    async def no_mcp():
        return {}
    monkeypatch.setattr(tools, "mcp_tools", no_mcp)
    viewer_specs, _ = await tools.tools_for(VIEWER)
    admin_specs, unavailable = await tools.tools_for(ADMIN)
    names = lambda specs: {s["function"]["name"] for s in specs}  # noqa: E731
    assert names(viewer_specs) == {"knowledge_search"}
    assert {"python_analysis", "view_audit_log"} <= names(admin_specs)
    assert "create_incident" in unavailable  # MCP down is reported, not hidden


async def test_invalid_arguments_rejected():
    c, _ = ctx(ANALYST)
    result = await execute_tool(c, "knowledge_search", {"query": "x"})
    assert not result["ok"] and result["error"].startswith("Invalid arguments")


async def test_search_then_python_analysis():
    c, _ = ctx(ANALYST)
    await execute_tool(c, "knowledge_search", {"query": "payments ledger connection pool", "document_types": ["incident"]})
    assert c.evidence
    code = "print(sorted({e['doc_id'] for e in data['evidence']}))"
    result = await execute_tool(c, "python_analysis", {"code": code})
    assert result["ok"] and "INC-" in result["result"]["output"]

    bad = await execute_tool(c, "python_analysis", {"code": "import os"})
    assert not bad["ok"] and "not allowed" in bad["error"]


async def test_tool_timeout_is_reported(monkeypatch):
    monkeypatch.setitem(resilience.FAULTS, "tool_timeout", True)
    monkeypatch.setattr(tools.settings, "tool_timeout_s", 0.1)
    c, _ = ctx(ANALYST)
    result = await execute_tool(c, "knowledge_search", {"query": "gateway timeout"})
    assert not result["ok"] and result["error"].startswith("Timed out")


async def test_mcp_down_degrades(monkeypatch):
    async def no_mcp():
        return {}
    monkeypatch.setattr(tools, "mcp_tools", no_mcp)
    c, _ = ctx(ANALYST)
    result = await execute_tool(c, "get_service", {"service": "crestpay-gateway"})
    assert not result["ok"] and "unavailable" in result["error"]
