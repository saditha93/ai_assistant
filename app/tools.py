import asyncio
import datetime as dt
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field, ValidationError

from app import memory
from app.auth import User
from app.config import settings
from app.observability import log
from app.resilience import FAULTS, mcp_breaker
from app.retrieval.search import hybrid_search
from app.sandbox import run_code

MAX_RESULT_CHARS = 4000
DocType = Literal["incident", "architecture", "runbook", "policy", "product_spec", "meeting_notes"]


@dataclass
class ToolContext:
    user: User
    store: BaseStore | None
    emit: Callable[[dict], None]
    evidence: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="What to search for")
    document_types: list[DocType] = Field(default_factory=list, description="Optional document type filter")
    since: dt.date | None = Field(None, description="Only documents created on or after this date")


class PythonAnalysisArgs(BaseModel):
    code: str = Field(
        max_length=4000,
        description="Python code to analyse the data gathered so far. The variable `data` holds "
                    "{'evidence': [document chunks with metadata], 'tool_results': [earlier tool outputs]}. "
                    "Counter, defaultdict, mean and median are available. No imports. Use print() for output.",
    )


class AuditLogArgs(BaseModel):
    limit: int = Field(20, ge=1, le=100)


async def _knowledge_search(ctx: ToolContext, args: KnowledgeSearchArgs) -> dict:
    filters = {"document_types": args.document_types, "since": args.since.isoformat() if args.since else None}
    result = await hybrid_search(args.query, ctx.user, filters=filters)
    ctx.evidence.extend(result["hits"])
    return {
        "mode": result["mode"],
        "results": [{"id": h["id"], "title": h["title"], "section": h["section"],
                     "created_date": h["created_date"], "text": h["text"][:600]} for h in result["hits"]],
    }


async def _python_analysis(ctx: ToolContext, args: PythonAnalysisArgs) -> dict:
    data = {"evidence": ctx.evidence, "tool_results": [r["result"] for r in ctx.results if r["ok"]]}
    result = await asyncio.to_thread(run_code, args.code, {"data": data}, 10)
    if not result["ok"]:
        raise ValueError(result["error"])
    return {"output": result["output"]}


async def _view_audit_log(ctx: ToolContext, args: AuditLogArgs) -> dict:
    return {"entries": await memory.read_audit(ctx.store, args.limit)} if ctx.store else {"entries": []}


LOCAL_TOOLS = {
    "knowledge_search": (KnowledgeSearchArgs, _knowledge_search,
                         "Search the bank's internal documents (policies, runbooks, architecture, incidents, "
                         "product specs, meeting notes). Returns passages with ids for citation."),
    "python_analysis": (PythonAnalysisArgs, _python_analysis,
                        "Run a short Python analysis (counts, averages, grouping) over the documents and tool "
                        "results gathered so far in this conversation turn."),
    "view_audit_log": (AuditLogArgs, _view_audit_log,
                       "Administrators only: read recent audit log entries (tool calls, denials, blocked requests)."),
}


_mcp_cache: dict = {"tools": {}, "loaded_at": 0.0}


async def mcp_tools(refresh: bool = False) -> dict[str, BaseTool]:
    """Tools published by the MCP server, cached for a minute. Empty if the server is down."""
    if not refresh and _mcp_cache["tools"] and time.monotonic() - _mcp_cache["loaded_at"] < 60:
        return _mcp_cache["tools"]
    client = MultiServerMCPClient({"enterprise": {"transport": "streamable_http", "url": settings.mcp_url}})
    try:
        tools = await mcp_breaker.call(client.get_tools, timeout=settings.tool_timeout_s)
    except Exception as exc:
        log.warning("mcp_unavailable", error=str(exc))
        _mcp_cache["tools"] = {}
        return {}
    _mcp_cache.update(tools={t.name: t for t in tools}, loaded_at=time.monotonic())
    return _mcp_cache["tools"]


def _check_mcp_args(tool: BaseTool, args: dict) -> None:
    """MCP tools validate on the server too; here we reject anything outside their schema
    before it leaves our process."""
    schema = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
    allowed = set(schema.get("properties", {}))
    unknown = set(args) - allowed
    if unknown:
        raise ValueError(f"unknown arguments: {sorted(unknown)}")
    missing = set(schema.get("required", [])) - set(args)
    if missing:
        raise ValueError(f"missing arguments: {sorted(missing)}")
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 1000:
            raise ValueError(f"argument '{key}' is too long")


async def _call_mcp(tool: BaseTool, args: dict) -> object:
    """MCP returns content blocks; ours are JSON text, so hand the agents plain data."""
    blocks = await mcp_breaker.call(tool.ainvoke, args, timeout=settings.tool_timeout_s)
    if not isinstance(blocks, list):
        return blocks
    parsed = []
    for block in blocks:
        text = block.get("text", "") if isinstance(block, dict) else str(block)
        try:
            parsed.append(json.loads(text))
        except ValueError:
            parsed.append(text)
    return parsed[0] if len(parsed) == 1 else parsed


async def tools_for(user: User) -> tuple[list, list[str]]:
    """Tool definitions to bind to the LLM for this user, plus names of tools that exist
    but are unavailable right now (so the answer can say so)."""
    specs = [
        {"type": "function", "function": {"name": name, "description": desc,
                                          "parameters": model.model_json_schema()}}
        for name, (model, _, desc) in LOCAL_TOOLS.items() if user.can_use(name)
    ]
    wanted_mcp = {"search_employees", "get_service", "search_incidents", "create_incident"} & user.tools
    unavailable = []
    if wanted_mcp:
        available = await mcp_tools()
        specs += [t for name, t in available.items() if name in wanted_mcp]
        unavailable = sorted(wanted_mcp - set(available))
    return specs, unavailable


def _shorten(value) -> object:
    text = json.dumps(value, default=str)
    return value if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "... [truncated]"


async def execute_tool(ctx: ToolContext, name: str, args: dict) -> dict:
    user = ctx.user
    ctx.emit({"type": "tool", "status": "call", "tool": name, "args": args})
    started = time.perf_counter()

    if not user.can_use(name):
        await memory.audit(ctx.store, user, "tool_denied", {"tool": name, "args": args})
        ctx.emit({"type": "tool", "status": "denied", "tool": name,
                  "reason": f"role '{user.role}' may not use this tool"})
        return {"tool": name, "args": args, "ok": False, "error": f"Not permitted for role '{user.role}'"}

    try:
        if name in LOCAL_TOOLS:
            model, fn, _ = LOCAL_TOOLS[name]
            parsed = model.model_validate(args)
            coro = fn(ctx, parsed)
        else:
            tools = await mcp_tools()
            if name not in tools:
                raise ConnectionError("enterprise MCP server is unavailable")
            tool = tools[name]
            _check_mcp_args(tool, args)
            coro = _call_mcp(tool, args)
        if FAULTS["tool_timeout"]:
            coro = _hang(coro)
        result = await asyncio.wait_for(coro, timeout=settings.tool_timeout_s)
        outcome = {"tool": name, "args": args, "ok": True, "result": _shorten(result)}
    except ValidationError as exc:
        outcome = {"tool": name, "args": args, "ok": False, "error": f"Invalid arguments: {exc.errors()[0]['msg']}"}
    except TimeoutError:
        outcome = {"tool": name, "args": args, "ok": False,
                   "error": f"Timed out after {settings.tool_timeout_s:.0f}s"}
    except Exception as exc:
        outcome = {"tool": name, "args": args, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    outcome["ms"] = round((time.perf_counter() - started) * 1000)
    ctx.results.append(outcome)
    await memory.audit(ctx.store, user, "tool_call", {"tool": name, "args": args, "ok": outcome["ok"]})
    ctx.emit({"type": "tool", "status": "result" if outcome["ok"] else "error", "tool": name,
              "ms": outcome["ms"], "error": outcome.get("error"),
              "preview": json.dumps(outcome.get("result"), default=str)[:300] if outcome["ok"] else None})
    return outcome


async def _hang(coro):
    coro.close()
    await asyncio.sleep(settings.tool_timeout_s + 5)
