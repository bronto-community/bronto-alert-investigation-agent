"""Audit-search tools for the identification phase."""

import json
from typing import Any

from bronto_sdk import BrontoClient
from bronto_sdk.domain import iso_to_ms, quote_value
from langchain_core.messages import AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from ...state import EVALUATION_START_KEYS, evaluation_start

# Bronto audit-trail dataset and the query fragments that isolate a
# monitor transitioning *into* an alert state (a triggered alert).
AUDIT_DATASET = ".audit-trail.events"
MONITOR_NAME_GROUP = '"service_payload.monitor_name"'
AUDIT_EVENTS_FILTER = (
    "\"event_info.type\"='MONITOR_STATE_CHANGE' "
    "AND \"service_payload.new_status\"='ALERT' "
    "AND \"service_payload.previous_status\" IN ('OK', 'WARN')"
)


def _audit_events_filter(pattern: str | None) -> str:
    """Build the ``where`` clause, optionally scoped by a name pattern.

    The pattern is appended last so the query engine applies the cheap,
    selective state-change filters before the name match.
    """
    if pattern:
        return f"{AUDIT_EVENTS_FILTER} AND {quote_value(pattern)}"
    return AUDIT_EVENTS_FILTER


class AlertTimeseries(BaseModel):
    """A time-series of the number of times a monitor has triggered in a time range."""

    monitor_name: str
    total: int
    counts: list[int]


def make_alert_timeseries_tool(client: BrontoClient) -> BaseTool:
    """Build the alert-timeseries tool bound to a Bronto client."""

    @tool
    def search_alert_timeseries(
        start: str, end: str, pattern: str | None = None
    ) -> list[AlertTimeseries]:
        """Return a 20-slice time series of the top alerts that triggered.

        The window runs from `start` to `end`, given as ISO-8601 UTC timestamps
        (e.g. '2026-07-15T00:00:00Z'). `pattern` optionally filters alert names
        and may contain `%` wildcards (0+ characters).
        """
        response = client.post(
            "/search",
            json={
                "async_enabled": False,
                "from": [AUDIT_DATASET],
                "groups": [MONITOR_NAME_GROUP],
                "num_of_slices": 20,
                "limit": 10,
                "select": ["count(*)"],
                "from_ts": iso_to_ms(start),
                "to_ts": iso_to_ms(end),
                "where": _audit_events_filter(pattern),
            },
        )
        return _adapt_timeseries(response)

    return search_alert_timeseries


def _adapt_timeseries(response: dict[str, Any]) -> list[AlertTimeseries]:
    """Adapt a /search response's groups_series into per-monitor series.

    Ordered by total triggers descending, so the top alerts come first.
    """
    series = [
        AlertTimeseries(
            monitor_name=group["name"],
            total=group["count"],
            counts=[slot["count"] for slot in group["timeseries"]],
        )
        for group in response.get("groups_series", [])
    ]
    return sorted(series, key=lambda s: s.total, reverse=True)


def make_audit_events_tool(client: BrontoClient) -> BaseTool:
    """Build the audit-event search tool bound to a Bronto client."""

    @tool(response_format="content_and_artifact")
    def search_audit_events(
        start: str, end: str, pattern: str | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return up to the 50 most recent triggered-alert audit events.

        The window runs from `start` to `end`, given as ISO-8601 UTC timestamps
        (e.g. '2026-07-15T00:00:00Z'). `pattern` optionally filters alert names
        and may contain `%` wildcards.
        """
        response = client.post(
            "/search",
            json={
                "async_enabled": False,
                "from": [AUDIT_DATASET],
                "select": ["*"],
                "most_recent_first": True,
                "from_ts": iso_to_ms(start),
                "to_ts": iso_to_ms(end),
                "where": _audit_events_filter(pattern),
                "limit": 50,
            },
        )
        return _adapt_events(response)

    return search_audit_events


_SERVICE_PAYLOAD_PREFIX = "service_payload."


def _adapt_events(response: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Adapt /search result rows into (content, artifact).

    ``content`` is the lean JSON the model reads; ``artifact`` is every
    ``service_payload`` field (prefix stripped), kept on the ToolMessage where
    the model never sees it. ``find_audit_row`` reads a row back out.

    ``triggered_at`` is the notification fire time (the event's top-level
    ``timestamp``); ``evaluation_timestamp`` is the monitor window's start, for
    context generation, read under either of its names (see ``evaluation_start``).
    """
    content: list[dict[str, Any]] = []
    artifact: list[dict[str, Any]] = []
    for event in response.get("result", []):
        payload = {
            key[len(_SERVICE_PAYLOAD_PREFIX) :]: value
            for key, value in event.items()
            if key.startswith(_SERVICE_PAYLOAD_PREFIX)
        }
        triggered_at = _format_fire_time(event["timestamp"])
        evaluation_timestamp = int(evaluation_start(payload))
        artifact.append(
            {
                **{k: v for k, v in payload.items() if k not in EVALUATION_START_KEYS},
                "evaluation_timestamp": evaluation_timestamp,
                "triggered_at": triggered_at,
            }
        )
        content.append(
            {
                "triggered_at": triggered_at,
                "monitor_name": payload.get("monitor_name"),
                "monitor_id": payload.get("monitor_id"),
            }
        )
    return json.dumps(content), artifact


def find_audit_row(
    messages: list[AnyMessage], monitor_id: str, triggered_at: str | None
) -> dict[str, Any] | None:
    """Find the full audit row behind an alert the model picked from ``content``.

    After a search, the agent's messages hold::

        AIMessage    search_audit_events(start=..., end=...)
        ToolMessage  content='[{"triggered_at": "2026-07-24T14:19:09Z",
                                "monitor_name": "OOM killer", "monitor_id": "mon-1"}]'
                     artifact=[{"monitor_id": "mon-1", "monitor_name": "OOM killer",
                                "triggered_at": "2026-07-24T14:19:09Z",
                                "evaluation_timestamp": 1784901720000, ...}]
        AIMessage    confirm_alert(candidates=[
                         {"id": "mon-1", "occurred_at": "2026-07-24T14:19:09Z", ...}])

    The model only ever saw ``content``, so it can only hand back what was there.
    This matches on both keys because a monitor can fire more than once: the id
    alone names the monitor, not the alert. No match means the model invented it.
    """
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        rows: list[dict[str, Any]] = message.artifact or []
        for row in rows:
            if (
                row.get("monitor_id") == monitor_id
                and row.get("triggered_at") == triggered_at
            ):
                return row
    return None


def _format_fire_time(timestamp: str) -> str:
    """Format the audit event's fire-time `timestamp` to whole-second ISO.

    The raw value carries nanosecond precision (e.g. '...:09.453550458Z') that
    fromisoformat can't parse, so truncate the fractional part.
    """
    return f"{timestamp.split('.', 1)[0].rstrip('Z')}Z"


def fetch_configured_playbook(client: BrontoClient, monitor_id: str) -> str | None:
    """Return the playbook configured for a monitor, or None if none is set."""
    return client.monitors.retrieve(monitor_id).ai_report_instructions
