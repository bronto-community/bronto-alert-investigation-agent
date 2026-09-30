"""Tests for the bolt handlers.

The handlers are free functions taking literal Slack payloads, so none of this
needs a Bolt dispatcher, a socket, or a workspace. The guards are what is
actually worth testing — the bodies are three lines each.
"""

import pytest
from slack_sdk.web.async_client import AsyncWebClient

from investigator.entrypoints.slack.blocks import format_action_id
from investigator.entrypoints.slack.bolt import (
    ACTION_PATTERN,
    build_bolt_app,
    handle_action,
    handle_command,
    handle_mention,
    handle_message,
    strip_mention,
)
from investigator.entrypoints.slack.runs import RunRegistry
from tests.conftest import confirm_alert_interrupt, mock_graph, until
from tests.entrypoints.slack.conftest import FakeSlackApi

_CANDIDATES = (
    {"id": "m-1", "label": "checkout errors"},
    {"id": "m-2", "label": "payment latency"},
)


def _asking(mocker, *then):
    """A graph that asks one question and, once answered, returns ``then``."""
    return mock_graph(
        mocker, {"__interrupt__": [confirm_alert_interrupt(*_CANDIDATES)]}, *then
    )


def _client(mocker, ts="1700000000.000100"):
    """A web client that enforces the real SDK's signatures."""
    client = mocker.create_autospec(AsyncWebClient, instance=True)
    client.chat_postMessage.return_value = {"ts": ts}
    return client


def _command_body(text="oom on checkout"):
    return {"channel_id": "C1", "user_id": "U1", "team_id": "T1", "text": text}


async def _live_run(registry):
    """Start a run and wait until it is asking its first question."""
    api = FakeSlackApi()
    handle = registry.start(
        api, team_id="T1", channel_id="C1", thread_ts="1.0", request_text="oom"
    )
    await until(lambda: handle.human.pending_seq == 1)
    return handle, api


async def _decision_after(handle, graph):
    """The decision the answered question produced, once the run has resumed."""
    await handle.task
    return graph.ainvoke.call_args_list[1].args[0].resume["decisions"][0]


# --- /investigate -------------------------------------------------------------


async def test_the_command_posts_a_root_message_and_starts_a_run(mocker):
    """A slash command cannot be used inside a thread, so it always mints its
    own root message and that ts becomes the run's thread."""
    registry = RunRegistry(_asking(mocker))
    client = _client(mocker)

    await handle_command(registry, client, _command_body())

    assert (
        "Investigating: oom on checkout"
        in client.chat_postMessage.call_args.kwargs["text"]
    )
    assert registry.get("C1", "1700000000.000100") is not None
    await registry.shutdown()


async def test_the_command_refuses_past_the_cap_without_posting(mocker):
    registry = RunRegistry(mock_graph(mocker), max_live=0)
    client = _client(mocker)

    await handle_command(registry, client, _command_body())

    client.chat_postMessage.assert_not_called()
    assert "already running" in client.chat_postEphemeral.call_args.kwargs["text"]


async def test_not_being_in_the_channel_gets_an_actionable_message(mocker):
    """The single most likely real-world failure, and a traceback would tell
    the user nothing they can act on."""
    from slack_sdk.errors import SlackApiError

    registry = RunRegistry(mock_graph(mocker))
    client = _client(mocker)
    client.chat_postMessage = mocker.AsyncMock(
        side_effect=SlackApiError("nope", {"error": "not_in_channel"})
    )

    await handle_command(registry, client, _command_body())

    assert "Invite me" in client.chat_postEphemeral.call_args.kwargs["text"]


async def test_any_other_slack_error_is_not_swallowed(mocker):
    from slack_sdk.errors import SlackApiError

    registry = RunRegistry(mock_graph(mocker))
    client = _client(mocker)
    client.chat_postMessage = mocker.AsyncMock(
        side_effect=SlackApiError("nope", {"error": "invalid_auth"})
    )

    with pytest.raises(SlackApiError):
        await handle_command(registry, client, _command_body())


# --- @mention -----------------------------------------------------------------


def test_the_mention_itself_is_stripped_from_the_description():
    assert strip_mention("<@U012AB3CD> what happened at 3pm") == "what happened at 3pm"


async def test_a_mention_runs_in_the_thread_it_happened_in(mocker):
    registry = RunRegistry(_asking(mocker))

    await handle_mention(
        registry,
        _client(mocker),
        {
            "channel": "C1",
            "ts": "9.9",
            "thread_ts": "1.0",
            "team": "T1",
            "text": "<@U1> what fired here?",
        },
    )

    assert registry.get("C1", "1.0") is not None
    await registry.shutdown()


async def test_a_mention_in_a_live_thread_does_not_start_a_second_run(mocker):
    """An @mention inside a live thread arrives twice — here and as a `message`.
    Without this guard you both answer the pending question and start a rival
    run in the same thread."""
    registry = RunRegistry(_asking(mocker))
    handle, _ = await _live_run(registry)

    await handle_mention(
        registry,
        _client(mocker),
        {"channel": "C1", "ts": "9.9", "thread_ts": "1.0", "text": "<@U1> hurry up"},
    )

    assert registry.get("C1", "1.0") is handle
    await registry.shutdown()


# --- the message handler, where the loop bug lives ----------------------------


async def test_a_thread_reply_answers_the_open_question(mocker):
    """Free text on this question sends the agent back to search."""
    graph = _asking(mocker, {"report": "done"})
    registry = RunRegistry(graph)
    handle, _ = await _live_run(registry)

    handle_message(
        registry, {"channel": "C1", "thread_ts": "1.0", "text": "try payments"}
    )

    decision = await _decision_after(handle, graph)
    assert decision["type"] == "respond"
    assert "try payments" in decision["message"]


@pytest.mark.parametrize(
    "event,why",
    [
        (
            {"channel": "C1", "thread_ts": "1.0", "text": "x", "bot_id": "B1"},
            "another bot posting in the thread",
        ),
        (
            {"channel": "C1", "thread_ts": "1.0", "text": "x", "app_id": "A1"},
            "an app posting in the thread",
        ),
        (
            {
                "channel": "C1",
                "thread_ts": "1.0",
                "text": "x",
                "subtype": "bot_message",
            },
            "a subtype: joins, edits, deletes, file shares",
        ),
        ({"channel": "C1", "text": "x"}, "top-level chatter with no thread"),
        (
            {"channel": "C1", "thread_ts": "9.9", "text": "x"},
            "a thread we never started",
        ),
        (
            {"channel": "C2", "thread_ts": "1.0", "text": "x"},
            "the right ts in the wrong human",
        ),
    ],
)
async def test_the_message_handler_ignores_everything_that_is_not_an_answer(
    mocker, event, why
):
    registry = RunRegistry(_asking(mocker))
    handle, _ = await _live_run(registry)
    submit = mocker.spy(handle.human, "submit")

    handle_message(registry, event)

    assert not submit.called, f"should have ignored {why}"
    await registry.shutdown()


async def test_a_reply_to_a_buttons_only_question_is_ignored(mocker):
    """Otherwise "thanks" resolves the Future, fails validation, and the whole
    question is re-posted at whoever said it."""
    from investigator.entrypoints.slack.thread import SlackHuman
    from investigator.human import Question

    registry = RunRegistry(_asking(mocker, {"report": "done"}))
    handle, _ = await _live_run(registry)
    # Swap in a human whose open question takes no free text.
    import asyncio

    human = SlackHuman(FakeSlackApi(), thread_ts="1.0", nonce="n")
    task = asyncio.create_task(
        human.ask(
            Question(action="pick", prompt="?", free_text="none", allow_skip=True)
        )
    )
    await until(lambda: human.pending_seq == 1)
    handle.human = human
    submit = mocker.spy(human, "submit")

    handle_message(registry, {"channel": "C1", "thread_ts": "1.0", "text": "thanks!"})

    submit.assert_not_called()
    task.cancel()
    await registry.shutdown()


# --- block actions ------------------------------------------------------------


def _action_body(action_id, *, thread_ts="1.0", channel="C1", selected=None):
    action = {"action_id": action_id}
    if selected is not None:
        action["selected_option"] = {"value": selected}
    return {
        "actions": [action],
        "channel": {"id": channel},
        "user": {"id": "U1"},
        "message": {"ts": "2.0", "thread_ts": thread_ts},
    }


async def test_a_click_answers_the_question_it_was_posted_for(mocker):
    graph = _asking(mocker, {"report": "done"})
    registry = RunRegistry(graph)
    handle, _ = await _live_run(registry)

    await handle_action(registry, _client(mocker), _action_body("ai:1:opt:1"))

    decision = await _decision_after(handle, graph)
    assert decision["edited_action"]["args"]["candidates"][0]["id"] == "m-2"


async def test_a_stale_click_is_refused_privately(mocker):
    """Ephemeral rather than a thread reply: a stale click is the clicker's
    problem and should not clutter everyone else's incident thread."""
    registry = RunRegistry(_asking(mocker))
    await _live_run(registry)
    client = _client(mocker)

    await handle_action(registry, client, _action_body("ai:99:opt:0"))

    assert "already been answered" in client.chat_postEphemeral.call_args.kwargs["text"]
    await registry.shutdown()


async def test_a_click_on_a_dead_run_says_so(mocker):
    registry = RunRegistry(mock_graph(mocker))
    client = _client(mocker)

    await handle_action(registry, client, _action_body("ai:1:opt:0", thread_ts="9.9"))

    assert "no longer running" in client.chat_postEphemeral.call_args.kwargs["text"]


async def test_a_select_carries_its_sequence_in_the_chosen_value(mocker):
    """One action_id serves every option in a select, so the seq rides along
    with the answer instead."""
    graph = _asking(mocker, {"report": "done"})
    registry = RunRegistry(graph)
    handle, _ = await _live_run(registry)

    await handle_action(
        registry, _client(mocker), _action_body("ai:1:select", selected="1:1")
    )

    decision = await _decision_after(handle, graph)
    assert decision["edited_action"]["args"]["candidates"][0]["id"] == "m-2"


async def test_an_action_that_is_not_ours_is_ignored(mocker):
    registry = RunRegistry(_asking(mocker))
    handle, _ = await _live_run(registry)
    submit = mocker.spy(handle.human, "submit")

    await handle_action(registry, _client(mocker), _action_body("some_other_app:go"))

    submit.assert_not_called()
    await registry.shutdown()


# --- wiring -------------------------------------------------------------------


def test_the_bolt_app_registers_every_listener(mocker):
    """Constructing an AsyncApp performs no network call — authorization is
    lazy middleware — so this is free. Four: command, mention, action, message."""
    app = build_bolt_app(RunRegistry(mock_graph(mocker)), token="xoxb-test")

    assert len(app._async_listeners) == 4


def test_the_action_listener_matches_the_ids_the_blocks_mint():
    """The pattern is written out in bolt.py and the prefix in blocks.py; if
    they drift, every click is silently dropped by the dispatcher."""
    assert ACTION_PATTERN.match(format_action_id(1, "opt", "0"))
    assert ACTION_PATTERN.match(format_action_id(1, "select"))
