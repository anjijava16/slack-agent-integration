"""
Tools the agent can call.

Each tool is a plain Python function with a type-annotated signature and a
docstring. The @tool decorator turns that into a JSON schema the LLM sees, so
the docstring is effectively part of the prompt — write it for the model.

The "backend systems" here (knowledge base, ticket system) are in-memory fakes
so the project runs with zero external dependencies. Swap them for real
clients (Confluence, Jira, ServiceNow, a vector store, an MCP server...) and the
graph doesn't change at all.
"""

from __future__ import annotations

import ast
import operator
import threading
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

# ---------------------------------------------------------------------------
# 1. Calculator — safe arithmetic (never eval() model output)
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** ( ) are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. "(1250 * 0.18) + 40".
    Use this for any math instead of doing it in your head."""
    try:
        result = _safe_eval(ast.parse(expression, mode="eval"))
    except Exception as exc:  # noqa: BLE001 - we want to return the error to the model
        return f"Error: could not evaluate {expression!r}: {exc}"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# 2. Clock
# ---------------------------------------------------------------------------
@tool
def get_current_time(tz: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone such as
    "America/New_York", "Asia/Kolkata" or "UTC"."""
    try:
        zone = ZoneInfo(tz)
    except ZoneInfoNotFoundError:
        return f"Error: unknown timezone {tz!r}. Use an IANA name like America/New_York."
    now = datetime.now(zone)
    return now.strftime("%A, %B %d, %Y %H:%M:%S %Z")


# ---------------------------------------------------------------------------
# 3. Knowledge base — a tiny stand-in for Confluence / SharePoint / a vector DB
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "id": "KB-001",
        "title": "VPN access",
        "body": "Install GlobalConnect from the Self Service portal. Sign in with SSO. "
        "If the VPN drops every few minutes, switch the protocol to IPSec in settings. "
        "VPN issues that last more than 30 minutes should be raised as a P2 ticket.",
    },
    {
        "id": "KB-002",
        "title": "Laptop replacement policy",
        "body": "Laptops are refreshed every 36 months. Broken screens or batteries are "
        "replaced within 2 business days; open a ticket with the asset tag. "
        "Loaners are available at the IT desk on floor 3.",
    },
    {
        "id": "KB-003",
        "title": "Paid time off (PTO)",
        "body": "Full-time employees accrue 1.75 PTO days per month (21 days per year). "
        "Up to 5 unused days roll over into the next year. Requests go through Workday "
        "and need manager approval at least 5 business days in advance.",
    },
    {
        "id": "KB-004",
        "title": "Production deploy freeze",
        "body": "No production deploys from the second Friday of December until January 2, "
        "and during the last 2 business days of each quarter. Emergency fixes need a P1 "
        "ticket and approval from the on-call engineering manager.",
    },
    {
        "id": "KB-005",
        "title": "On-call rotation",
        "body": "Each team runs a weekly on-call rotation starting Monday 10:00 local time. "
        "On-call engineers must acknowledge P1 pages within 15 minutes and P2 pages within 1 hour.",
    },
    {
        "id": "KB-006",
        "title": "Password reset",
        "body": "Reset your SSO password at the Self Service portal. Passwords expire every 90 days, "
        "need at least 14 characters, and can't reuse the last 10 passwords.",
    },
]


@tool
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the internal company knowledge base (IT, HR and engineering policies).
    Use it before answering any question about company policy, VPN, laptops, PTO,
    deploy freezes, on-call or passwords. Returns matching articles with their KB ids."""
    terms = {t for t in query.lower().replace("?", " ").split() if len(t) > 2}
    scored = []
    for doc in KNOWLEDGE_BASE:
        text = f"{doc['title']} {doc['body']}".lower()
        score = sum(text.count(t) for t in terms) + 3 * sum(t in doc["title"].lower() for t in terms)
        if score:
            scored.append((score, doc))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored:
        return "No knowledge base articles matched. Tell the user you couldn't find a policy for this."
    return "\n\n".join(f"[{d['id']}] {d['title']}\n{d['body']}" for _, d in scored[: max(1, min(top_k, 5))])


# ---------------------------------------------------------------------------
# 4. Ticketing — a stand-in for Jira / ServiceNow
# ---------------------------------------------------------------------------
class TicketStore:
    """Thread-safe in-memory ticket store. Replace with a real API client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tickets: dict[str, dict] = {}

    def create(self, title: str, description: str, priority: str, requester: str) -> dict:
        ticket = {
            "id": f"TCK-{uuid.uuid4().hex[:6].upper()}",
            "title": title,
            "description": description,
            "priority": priority,
            "requester": requester,
            "status": "open",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with self._lock:
            self._tickets[ticket["id"]] = ticket
        return ticket

    def get(self, ticket_id: str) -> dict | None:
        with self._lock:
            return self._tickets.get(ticket_id.strip().upper())

    def list_open(self) -> list[dict]:
        with self._lock:
            return [t for t in self._tickets.values() if t["status"] == "open"]

    def clear(self) -> None:
        with self._lock:
            self._tickets.clear()


TICKETS = TicketStore()


@tool
def create_ticket(
    title: str,
    description: str,
    priority: Literal["P1", "P2", "P3", "P4"],
    # InjectedState: the value is read from the graph state at run time and is
    # NOT part of the schema the LLM sees. The model can't spoof who the
    # requester is — it always comes from the verified Slack user id.
    requester: Annotated[str, InjectedState("user_id")],
) -> str:
    """Open a support ticket. Only call this when the user actually asks for a
    ticket or clearly reports a problem that needs one. P1 = outage, P2 = major
    impact, P3 = normal, P4 = low/request. Returns the new ticket id."""
    ticket = TICKETS.create(title=title, description=description, priority=priority, requester=requester)
    return f"Created {ticket['id']} ({ticket['priority']}) '{ticket['title']}' for requester {requester}."


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up a support ticket by id, e.g. TCK-1A2B3C."""
    ticket = TICKETS.get(ticket_id)
    if not ticket:
        return f"No ticket found with id {ticket_id!r}."
    return (
        f"{ticket['id']} | {ticket['priority']} | {ticket['status']} | {ticket['title']} | "
        f"requested by {ticket['requester']} at {ticket['created_at']}"
    )


@tool
def list_open_tickets() -> str:
    """List every open support ticket."""
    open_tickets = TICKETS.list_open()
    if not open_tickets:
        return "There are no open tickets."
    return "\n".join(f"{t['id']} | {t['priority']} | {t['title']} | {t['requester']}" for t in open_tickets)


ALL_TOOLS = [
    calculator,
    get_current_time,
    search_knowledge_base,
    create_ticket,
    get_ticket_status,
    list_open_tickets,
]
