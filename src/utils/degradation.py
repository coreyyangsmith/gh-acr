"""ContextVar-backed tracker for soft pipeline degradations.

Soft degradations (prompt truncation, JSON/heuristic fallbacks, unclear
verdicts) complete without raising. Call ``record_degradation`` at fallback
sites; ``run_all`` checks ``has_degradations`` after each method unit, records
a ledger/failures entry, stamps identifiable flags on CSV rows
(``soft_degraded``, ``degradation_category``, ``num_degradations``), and still
writes those rows. Only hard exceptions omit CSV rows.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Categories used across the pipeline (see plan).
DEGRADATION_CATEGORIES = frozenset(
    {
        "prompt_truncation",
        "json_parse_fallback",
        "plan_schema_fallback",
        "llm_unavailable_heuristic",
        "unclear_verdict_fallback",
    }
)

_events: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "ghacr_degradation_events", default=None
)


def clear_degradations() -> None:
    """Reset the degradation event list for the current context."""
    _events.set([])


def record_degradation(
    category: str,
    reason: str,
    *,
    detail: str | None = None,
    node: str | None = None,
    file: str | None = None,
) -> None:
    """Append one degradation event to the current context list.

    If the tracker was never cleared in this context, events are still
    recorded into a fresh list so instrumentation is safe outside ``run_all``.
    """
    events = _events.get()
    if events is None:
        events = []
        _events.set(events)
    event: dict[str, Any] = {
        "category": category,
        "reason": reason,
    }
    if detail is not None:
        event["detail"] = detail
    if node is not None:
        event["node"] = node
    if file is not None:
        event["file"] = file
    events.append(event)


def get_degradations() -> list[dict[str, Any]]:
    """Return a copy of degradation events for the current context."""
    events = _events.get()
    if not events:
        return []
    return [dict(e) for e in events]


def has_degradations() -> bool:
    """True if at least one degradation was recorded in this context."""
    events = _events.get()
    return bool(events)


def primary_degradation_category(
    events: list[dict[str, Any]] | None = None,
) -> str | None:
    """Return the category of the first event (or None if empty)."""
    items = events if events is not None else get_degradations()
    if not items:
        return None
    return str(items[0].get("category") or "other")


__all__ = [
    "DEGRADATION_CATEGORIES",
    "clear_degradations",
    "record_degradation",
    "get_degradations",
    "has_degradations",
    "primary_degradation_category",
]
