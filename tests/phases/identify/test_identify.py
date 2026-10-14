"""Tests for "identify" part of the agent."""

from datetime import UTC, datetime

from langchain_core.messages import AIMessage

from investigator.phases.identify.confirm import CONFIRM_ALERT, CONFIRM_PLAYBOOK
from investigator.phases.identify.identify import (
    SYSTEM_PROMPT,
    ForceToolChoiceMiddleware,
    build_identification_node,
    make_playbook_lookup,
    system_prompt,
)
from investigator.settings import Deps
from tests.conftest import a_monitor


def _deps(mocker, model):
    return Deps(client=mocker.Mock(), model=model, investigation_tools=[])


def test_playbook_lookup_reads_the_id_off_the_grounded_row(mocker):
    """The monitor id comes from the confirmed audit row on the triggered alert, not
    from a model argument, so there is nothing here to hallucinate."""
    client = mocker.Mock()
    client.monitors.retrieve.return_value = a_monitor(
        ai_report_instructions="check redis"
    )
    deps = Deps(client=client, model=None, investigation_tools=[])
    triggered_alert = {"id": "mon-1", "data": {"monitor_id": "mon-1"}}

    assert make_playbook_lookup(deps)(triggered_alert) == "check redis"
    client.monitors.retrieve.assert_called_once_with("mon-1")


def test_playbook_lookup_tolerates_an_alert_without_a_monitor_id(mocker):
    """A triggered alert may not carry a monitor id; that is a missing playbook, not a
    crash."""
    assert make_playbook_lookup(_deps(mocker, None))({"id": "x"}) is None


async def test_the_model_is_bound_with_both_searches_and_both_questions(
    mocker, make_model
):
    """Read off what the model was actually offered on its first turn, rather
    than off the agent's internals."""
    model = make_model([AIMessage(content="not sure")])

    await build_identification_node(_deps(mocker, model))({"request_text": "oom"})

    bound = {
        t["function"]["name"] if isinstance(t, dict) else t.name
        for t in model.bound_tools
    }
    assert bound == {
        "search_alert_timeseries",
        "search_audit_events",
        CONFIRM_ALERT,
        CONFIRM_PLAYBOOK,
    }


def test_system_prompt_injects_todays_date():
    """The model gets today's date as ground truth for relative time phrases."""
    rendered = system_prompt("be an SRE", datetime(2026, 7, 24, tzinfo=UTC))

    assert "be an SRE" in rendered
    assert "2026-07-24" in rendered
    assert "ground truth" in rendered


def test_monitor_prompt_directs_the_model_to_both_hitl_tools():
    """The agent only progresses via the two human-in-the-loop tools, and the
    playbook one must be called on its own to end the phase."""
    rendered = system_prompt(SYSTEM_PROMPT, datetime(2026, 7, 24, tzinfo=UTC))

    assert CONFIRM_ALERT in rendered
    assert CONFIRM_PLAYBOOK in rendered
    assert "not batched" in rendered


async def test_a_model_that_cannot_be_forced_is_never_asked_to(mocker, make_model):
    """Bedrock rejects tool_choice "any" outright for some models, so for them
    identification leans on its system prompt rather than forcing."""
    model = make_model([AIMessage(content="not sure")], tool_choices=("auto",))

    await build_identification_node(_deps(mocker, model))({"request_text": "oom"})

    assert model.call_kwargs[-1].get("tool_choice") is None


async def test_a_model_that_can_be_forced_is(mocker, make_model):
    model = make_model([AIMessage(content="not sure")])

    await build_identification_node(_deps(mocker, model))({"request_text": "oom"})

    assert model.call_kwargs[-1]["tool_choice"] == "any"


async def test_force_tool_choice_requires_a_tool_every_turn(mocker):
    """Without a response_format nothing forces tool use, and a single prose
    turn would end the phase instead of looping back to search. The async hook
    is the one the graph drives."""
    request = mocker.Mock()
    handler = mocker.AsyncMock()

    await ForceToolChoiceMiddleware().awrap_model_call(request, handler)

    request.override.assert_called_once_with(tool_choice="any")
    handler.assert_awaited_once_with(request.override.return_value)
