"""The investigation state: everything that crosses between phases."""

from __future__ import annotations

import json
from typing import Any, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InvestigationState(TypedDict):
    """The whole run's state: only what crosses between phases."""

    request_text: str | None
    triggered_alert: dict[str, Any] | None
    playbook: str | None
    alert_context: AlertContext | None
    findings: dict[str, Any] | None
    report: str | None


class TriggeredAlert(BaseModel):
    """What the investigation is about.

    Produced two ways: by the identification phase's confirm tool,
    or by validating an external event payload.

    ``data`` is the audit-trail event the context-generation phase consumes.
    """

    id: str
    label: str
    occurred_at: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class AlertContext(TypedDict):
    """The handoff from context generation into investigation and report."""

    window: Window
    playbook: str | None
    detail: dict[str, Any]


class Window(TypedDict):
    """An absolute epoch-ms range."""

    start_ms: int
    end_ms: int


def state_for_request(request_text: str) -> InvestigationState:
    """Build a fresh state for a natural-language investigation request."""
    return {
        "request_text": request_text,
        "triggered_alert": None,
        "playbook": None,
        "alert_context": None,
        "findings": None,
        "report": None,
    }


def state_for_event(
    payload: dict[str, Any], playbook: str | None = None
) -> InvestigationState:
    """Build a state for a run that already knows what it is investigating.

    Raises ``pydantic.ValidationError`` for a payload the context builder could
    not work from, so a malformed webhook fails at the edge rather than
    part-way through a run that has already spent model calls.
    """
    event = MonitorEventPayload.model_validate(payload)
    triggered_alert = TriggeredAlert(
        id=event.monitor_id,
        label=event.monitor_name or event.monitor_id,
        occurred_at=event.triggered_at,
        data=event.model_dump(exclude_none=True),
    )
    return {
        "request_text": None,
        "triggered_alert": triggered_alert.model_dump(mode="json"),
        "playbook": playbook,
        "alert_context": None,
        "findings": None,
        "report": None,
    }


# Bronto renamed `evaluation_timestamp` to `evaluation_window_start_ts`.
EVALUATION_START_KEYS = ("evaluation_window_start_ts", "evaluation_timestamp")


def evaluation_start(payload: dict[str, Any]) -> Any:
    """The monitored window's start, under whichever name the event uses."""
    for key in EVALUATION_START_KEYS:
        if key in payload:
            return payload[key]
    raise KeyError(EVALUATION_START_KEYS[-1])


class MonitorEventPayload(BaseModel):
    """A monitor-alert event body, validated down to what context generation needs."""

    model_config = ConfigDict(extra="allow")

    monitor_id: str
    evaluation_timestamp: int
    monitor_name: str | None = None
    triggered_at: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _canonical_names(cls, data: Any) -> Any:
        """Carry the monitor and the window start under the names the run reads."""
        if not isinstance(data, dict):
            return data
        payload = cast(dict[str, Any], data)
        resource = payload.get("resource")
        if isinstance(resource, dict):
            payload = {
                **{k: v for k, v in payload.items() if k != "resource"},
                **{
                    f"{_RESOURCE_PREFIX}{k}": v
                    for k, v in cast(dict[str, Any], resource).items()
                },
            }
        payload = with_monitor_names(payload)
        try:
            start = evaluation_start(payload)
        except KeyError:
            return payload  # left for the field's own "required" error
        rest = {k: v for k, v in payload.items() if k not in EVALUATION_START_KEYS}
        return {**rest, "evaluation_timestamp": start}


# Audit schema 1.2.0 moved `monitor_id` and `monitor_name` under `resource`. The
# old names stop being emitted after a deprecation window; old events keep them.
MONITOR_KEYS = {
    "monitor_id": ("resource.id", "monitor_id"),
    "monitor_name": ("resource.name", "monitor_name"),
}
_RESOURCE_PREFIX = "resource."


def with_monitor_names(payload: dict[str, Any]) -> dict[str, Any]:
    """Carry the monitor's id and name under the names the run reads."""
    rest = {k: v for k, v in payload.items() if not k.startswith(_RESOURCE_PREFIX)}
    for name, keys in MONITOR_KEYS.items():
        value = next((payload[key] for key in keys if key in payload), None)
        if value is not None:
            rest[name] = value
    return rest


def alert_context_text(alert_context: AlertContext | None) -> str:
    """Render the alert context for the model.

    Flattens ``detail`` up beside the promoted ``playbook`` key, so the model
    sees one document rather than a wrapper it has to look inside. The window
    stays out of it: searches are already scoped to it in code, and the prompt
    tells the model not to set time ranges.
    """
    if not alert_context:
        return json.dumps(alert_context, default=str)
    return json.dumps(
        {
            "playbook": alert_context.get("playbook"),
            **(alert_context.get("detail") or {}),
        },
        default=str,
    )
