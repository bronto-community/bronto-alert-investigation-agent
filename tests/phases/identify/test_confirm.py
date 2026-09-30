"""Tests for phase 1's confirm tools and the just-in-time playbook lookup."""

import logging

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import ToolRuntime

from investigator.phases.identify.confirm import (
    CONFIRM_PLAYBOOK,
    Candidate,
    PlaybookLookupMiddleware,
    confirm_playbook,
    make_confirm_alert_tool,
)

CANDIDATE = Candidate(
    id="mon-1", label="foo bar baz", occurred_at="2026-07-24T14:19:09Z"
)

confirm_alert = make_confirm_alert_tool(description="confirm the alert")
ROW = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}


def _artifact_message(*rows):
    return ToolMessage(content="[]", tool_call_id="t1", artifact=list(rows))


# The confirmed alert as the tool leaves it on the agent's state.
_TRIGGERED_ALERT = {
    "id": "mon-1",
    "label": "foo bar baz",
    "occurred_at": "2026-07-24T14:19:09Z",
    "data": ROW,
}


def _runtime(messages=()):
    """A ToolRuntime as ToolNode injects one, for calling a tool body directly."""
    return ToolRuntime(
        state={"messages": list(messages)},
        context=None,
        config={},
        stream_writer=lambda _chunk: None,
        tool_call_id="c1",
        store=None,
    )


# --- grounding ----------------------------------------------------------------


def test_confirm_alert_writes_the_full_triggered_event():
    """The triggered alert carries the whole audit row, not the lean candidate summary —
    the context phase needs evaluation_timestamp, which the summary drops."""
    command = confirm_alert.func(
        candidates=[CANDIDATE], runtime=_runtime([_artifact_message(ROW)])
    )

    triggered_alert = command.update["triggered_alert"]
    assert triggered_alert["id"] == CANDIDATE.id
    assert triggered_alert["data"] == ROW
    assert isinstance(command.update["messages"][0], ToolMessage)


def test_confirm_alert_rejects_an_ungrounded_candidate(caplog):
    """A confirmed candidate must be grounded in a retrieved artifact; failing
    to link back means the selection was invented rather than found."""
    with caplog.at_level(logging.ERROR, logger="investigator.phases.identify.confirm"):
        with pytest.raises(ValueError, match="not found in retrieved artifacts"):
            confirm_alert.func(
                candidates=[CANDIDATE], runtime=_runtime([_artifact_message()])
            )

    assert "not in any retrieved artifact" in caplog.text


def test_confirm_alert_refuses_an_unnarrowed_proposal():
    """The interrupt narrows a multi-candidate proposal to the user's pick
    before the body runs, so more than one here means it was bypassed."""
    two = [CANDIDATE, CANDIDATE.model_copy(update={"id": "mon-2"})]

    with pytest.raises(ValueError, match="narrowed to one candidate"):
        confirm_alert.func(candidates=two, runtime=_runtime())


def test_confirm_playbook_writes_the_playbook():
    """The playbook tool is the phase terminator and simply records the choice."""
    command = confirm_playbook.func(playbook="check redis", runtime=_runtime())

    assert command.update["playbook"] == "check redis"


def test_confirm_playbook_accepts_a_skip():
    """Skipping is an explicit null playbook, not an absent one."""
    assert (
        confirm_playbook.func(playbook=None, runtime=_runtime()).update["playbook"]
        is None
    )


def test_confirm_playbook_ends_the_phase():
    """return_direct is what makes this the phase's exit rather than another
    loop back to the model."""
    assert confirm_playbook.return_direct is True


# --- the just-in-time playbook lookup -----------------------------------------


def _playbook_turn():
    return AIMessage(
        content="",
        tool_calls=[{"name": CONFIRM_PLAYBOOK, "args": {"playbook": None}, "id": "p1"}],
    )


async def test_playbook_lookup_hands_the_lookup_the_triggered_alert():
    """The monitor id comes from the confirmed audit row, not a model argument,
    so there is nothing here to hallucinate."""
    seen = []

    def lookup(triggered_alert):
        seen.append(triggered_alert)
        return "check redis"

    state = {"messages": [_playbook_turn()], "triggered_alert": _TRIGGERED_ALERT}

    delta = await PlaybookLookupMiddleware(lookup).aafter_model(state, None)

    assert seen == [_TRIGGERED_ALERT]
    assert delta == {"configured_playbook": "check redis"}


async def test_playbook_lookup_skips_other_turns(mocker):
    """A search turn must not trigger a monitor fetch."""
    lookup = mocker.Mock()
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"name": "search_audit_events", "args": {}, "id": "s1"}],
            )
        ]
    }

    assert await PlaybookLookupMiddleware(lookup).aafter_model(state, None) is None
    lookup.assert_not_called()


async def test_playbook_lookup_without_a_confirmed_alert(mocker):
    """Nothing to look up before a triggered alert is grounded."""
    lookup = mocker.Mock()
    state = {"messages": [_playbook_turn()], "triggered_alert": None}

    delta = await PlaybookLookupMiddleware(lookup).aafter_model(state, None)

    assert delta == {"configured_playbook": None}
    lookup.assert_not_called()


def test_playbook_lookup_implements_the_sync_hook_too(mocker):
    """Both hooks are implemented so the middleware works either way."""
    state = {"messages": [_playbook_turn()], "triggered_alert": _TRIGGERED_ALERT}

    delta = PlaybookLookupMiddleware(lambda triggered_alert: "check redis").after_model(
        state, None
    )

    assert delta == {"configured_playbook": "check redis"}
