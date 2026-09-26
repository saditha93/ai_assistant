import datetime as dt
import json
import os
from pathlib import Path
from typing import Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

DATA = Path(__file__).resolve().parent.parent / "data" / "mcp"
EMPLOYEES = json.loads((DATA / "employees.json").read_text())
SERVICES = json.loads((DATA / "services.json").read_text())
INCIDENTS = json.loads((DATA / "incidents.json").read_text())

mcp = FastMCP(
    "crestline-enterprise",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

Severity = Literal["SEV1", "SEV2", "SEV3"]


@mcp.tool()
def search_employees(query: str = "", department: str | None = None, limit: int = 10) -> list[dict]:
    """Find employees by name, title, email or the service they are on call for."""
    q = query.lower().strip()
    matches = [
        e for e in EMPLOYEES
        if (not department or e["department"] == department.lower())
        and (not q or q in " ".join([e["name"], e["title"], e["email"], *e["on_call_for"]]).lower())
    ]
    return matches[: max(1, min(limit, 50))]


@mcp.tool()
def get_service(service: str) -> dict:
    """Service catalog entry: owner team, tier, status, on-call engineer, dependencies and runbook."""
    entry = next((s for s in SERVICES if service.lower() in (s["service"], s["display_name"].lower())), None)
    if entry is None:
        return {"error": f"Unknown service '{service}'", "known_services": [s["service"] for s in SERVICES]}
    on_call = next((e for e in EMPLOYEES if e["employee_id"] == entry["on_call_employee_id"]), None)
    return {**entry, "on_call": on_call and {k: on_call[k] for k in ("name", "email", "phone_ext")}}


@mcp.tool()
def search_incidents(
    service: str | None = None,
    severity: Severity | None = None,
    status: Literal["open", "monitoring", "resolved"] | None = None,
    root_cause_category: str | None = None,
    since: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Incident records, newest first. `since` is an ISO date (YYYY-MM-DD)."""
    rows = [
        i for i in INCIDENTS
        if (not service or i["service"] == service)
        and (not severity or i["severity"] == severity)
        and (not status or i["status"] == status)
        and (not root_cause_category or i["root_cause_category"] == root_cause_category)
        and (not since or i["opened_at"][:10] >= since)
    ]
    rows.sort(key=lambda i: i["opened_at"], reverse=True)
    return rows[: max(1, min(limit, 50))]


@mcp.tool()
def create_incident(title: str, service: str, severity: Severity, description: str) -> dict:
    """Open a new incident record. This is a write action; the assistant asks a human to approve it first."""
    if service not in {s["service"] for s in SERVICES}:
        return {"error": f"Unknown service '{service}'"}
    year = dt.date.today().year
    number = 1 + sum(1 for i in INCIDENTS if i["incident_id"].startswith(f"INC-{year}-"))
    record = {
        "incident_id": f"INC-{year}-{number + 100:03d}", "title": title[:200], "service": service,
        "severity": severity, "status": "open", "opened_at": dt.datetime.now().isoformat(timespec="seconds"),
        "resolved_at": None, "root_cause_category": None, "customers_affected": 0, "duration_minutes": 0,
        "description": description[:2000],
    }
    INCIDENTS.append(record)
    return record


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
