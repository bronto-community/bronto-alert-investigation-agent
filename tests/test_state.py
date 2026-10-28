"""Tests for the top-level orchestrator state."""

import json

import pytest
from pydantic import ValidationError

from investigator.state import (
    InvestigationState,
    evaluation_start,
    state_for_event,
    state_for_request,
)

_EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}


def test_a_request_run_seeds_every_channel_with_only_the_request():
    state = state_for_request("checkout errors")

    assert set(state) == set(InvestigationState.__annotations__)
    assert state == {
        "request_text": "checkout errors",
        "triggered_alert": None,
        "playbook": None,
        "alert_context": None,
        "findings": None,
        "report": None,
    }


def test_an_event_run_seeds_every_channel_with_a_triggered_alert():
    state = state_for_event(_EVENT, playbook="check redis")

    assert set(state) == set(InvestigationState.__annotations__)
    assert state["request_text"] is None
    assert state["triggered_alert"]["id"] == "mon-1"
    assert state["triggered_alert"]["data"] == _EVENT
    assert state["playbook"] == "check redis"


def test_the_state_is_plain_json():
    state = state_for_event(_EVENT)

    assert json.loads(json.dumps(state)) == state


# --- the renamed window start -------------------------------------------------

_NEW = 1784901780000


def test_the_renamed_window_start_is_read_under_the_name_the_run_uses():
    """Bronto renamed `evaluation_timestamp` to `evaluation_window_start_ts`.
    The event is normalized at the edge; context generation still reads
    `evaluation_timestamp`."""
    event = {"monitor_id": "mon-1", "evaluation_window_start_ts": _NEW}

    data = state_for_event(event)["triggered_alert"]["data"]

    assert data == {"monitor_id": "mon-1", "evaluation_timestamp": _NEW}


def test_the_new_name_wins_when_an_event_carries_both():
    event = {**_EVENT, "evaluation_window_start_ts": _NEW}

    data = state_for_event(event)["triggered_alert"]["data"]

    assert data["evaluation_timestamp"] == _NEW
    assert "evaluation_window_start_ts" not in data


def test_an_event_with_neither_name_is_refused():
    with pytest.raises(ValidationError):
        state_for_event({"monitor_id": "mon-1"})


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"evaluation_window_start_ts": _NEW}, _NEW),
        ({"evaluation_timestamp": 1}, 1),
        ({"evaluation_window_start_ts": _NEW, "evaluation_timestamp": 1}, _NEW),
    ],
)
def test_evaluation_start_prefers_the_new_name(payload, expected):
    assert evaluation_start(payload) == expected


def test_evaluation_start_raises_when_neither_name_is_present():
    with pytest.raises(KeyError):
        evaluation_start({"monitor_id": "mon-1"})


# --- the monitor under `resource` ---------------------------------------------


def test_a_replayed_1_2_0_payload_is_read_under_the_names_the_run_uses():
    """Audit schema 1.2.0 moved the monitor's id and name under `resource`.
    A replayed `service_payload` is normalized at the edge, like the window start."""
    event = {
        "resource": {"type": "MONITOR", "id": "mon-1", "name": "foo bar baz"},
        "evaluation_timestamp": 1784901720000,
    }

    alert = state_for_event(event)["triggered_alert"]

    assert alert["id"] == "mon-1"
    assert alert["label"] == "foo bar baz"
    assert alert["data"] == {
        "monitor_id": "mon-1",
        "monitor_name": "foo bar baz",
        "evaluation_timestamp": 1784901720000,
    }


def test_the_resource_wins_when_an_event_carries_both():
    event = {**_EVENT, "resource": {"id": "mon-2", "name": "new name"}}

    data = state_for_event(event)["triggered_alert"]["data"]

    assert data["monitor_id"] == "mon-2"
    assert data["monitor_name"] == "new name"
    assert "resource" not in data


def test_an_event_with_neither_monitor_id_is_refused():
    with pytest.raises(ValidationError):
        state_for_event({"resource": {"name": "no id"}, "evaluation_timestamp": 1})
