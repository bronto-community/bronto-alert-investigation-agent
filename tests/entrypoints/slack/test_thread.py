"""Tests for answering a question that arrives from somewhere else entirely.

The interesting property is that `ask` blocks on a Future which a completely
separate code path resolves, possibly much later. These tests drive both sides
on one event loop, so the correlation is checked without a clock or a network.
"""

import asyncio

import pytest

from investigator.entrypoints.slack.thread import QuestionTimeout, SlackHuman
from investigator.human import Answer, Event, Question
from investigator.phases.identify.questions import (
    confirm_alert_question,
    playbook_question,
)
from tests.conftest import until

_C0 = {"id": "m-1", "label": "checkout errors", "occurred_at": "2026-07-24T14:02:00Z"}
_C1 = {"id": "m-2", "label": "payment latency", "occurred_at": "2026-07-24T09:30:00Z"}


def _alert_question(*candidates) -> Question:
    return Question.model_validate(
        confirm_alert_question({"args": {"candidates": list(candidates)}}, {}, None)
    )


def _playbook(configured=None) -> Question:
    return Question.model_validate(
        playbook_question({}, {"configured_playbook": configured}, None)
    )


def _channel(api, **kwargs) -> SlackHuman:
    return SlackHuman(api, thread_ts="1700000000.000100", nonce="n1", **kwargs)


# --- the happy path -----------------------------------------------------------


async def test_ask_posts_the_question_and_waits(slack_api):
    human = _channel(slack_api)

    task = asyncio.create_task(human.ask(_alert_question(_C0, _C1)))
    await until(lambda: human.pending_seq == 1)

    assert len(slack_api.posts) == 1
    assert not task.done(), "ask must not return before anybody answers"
    task.cancel()


async def test_a_click_resolves_the_waiting_ask(slack_api):
    human = _channel(slack_api)
    task = asyncio.create_task(human.ask(_alert_question(_C0, _C1)))
    await until(lambda: human.pending_seq == 1)

    assert human.submit(seq=1, option_id="1") is None

    assert await task == Answer(option_id="1")


async def test_a_thread_reply_resolves_it_too(slack_api):
    human = _channel(slack_api)
    task = asyncio.create_task(human.ask(_alert_question(_C0)))
    await until(lambda: human.pending_seq == 1)

    human.submit(seq=1, text="no, try the payment service")

    assert (await task).text == "no, try the payment service"


async def test_skip_resolves_as_a_skip(slack_api):
    human = _channel(slack_api)
    task = asyncio.create_task(human.ask(_playbook("check redis")))
    await until(lambda: human.pending_seq == 1)

    human.submit(seq=1, skip=True)

    assert (await task).skip is True


# --- the stale click, which a terminal never has to worry about ---------------


async def test_a_click_on_a_superseded_question_is_refused(slack_api):
    """The failure mode a chat surface has and a terminal does not: the old
    message is still on screen with its buttons live."""
    human = _channel(slack_api)
    first = asyncio.create_task(human.ask(_alert_question(_C0, _C1)))
    await until(lambda: human.pending_seq == 1)
    human.submit(seq=1, option_id="0")
    await first

    second = asyncio.create_task(human.ask(_playbook()))
    await until(lambda: human.pending_seq == 2)

    refusal = human.submit(seq=1, option_id="0")

    assert refusal == "that question has already been answered"
    second.cancel()


async def test_a_second_click_on_the_same_question_is_refused(slack_api):
    """Two people click at once; the first wins, which is the 'anyone in the
    thread' rule made concrete."""
    human = _channel(slack_api)
    task = asyncio.create_task(human.ask(_alert_question(_C0, _C1)))
    await until(lambda: human.pending_seq == 1)

    assert human.submit(seq=1, option_id="0") is None
    assert (
        human.submit(seq=1, option_id="1") == "that question has already been answered"
    )

    assert (await task).option_id == "0"


async def test_answering_when_nothing_was_asked_is_refused(slack_api):
    assert _channel(slack_api).submit(seq=1, text="hello?") == (
        "there is no question waiting in this thread"
    )


# --- what counts as an answer -------------------------------------------------


async def test_a_buttons_only_question_does_not_accept_thread_text(slack_api):
    """Otherwise someone typing "thanks" resolves the Future, fails validation,
    and gets the whole question re-posted at them."""
    human = _channel(slack_api)
    question = Question(action="pick", prompt="?", free_text="none", allow_skip=True)
    task = asyncio.create_task(human.ask(question))
    await until(lambda: human.pending_seq == 1)

    assert human.accepts_text is False
    task.cancel()


async def test_a_question_inviting_a_reply_accepts_text(slack_api):
    human = _channel(slack_api)
    task = asyncio.create_task(human.ask(_alert_question(_C0)))
    await until(lambda: human.pending_seq == 1)

    assert human.accepts_text is True
    task.cancel()


async def test_nothing_accepts_text_while_the_run_is_thinking(slack_api):
    assert _channel(slack_api).accepts_text is False


# --- validation, re-asked in thread -------------------------------------------


async def test_an_unusable_answer_is_re_asked_without_resuming_the_graph(slack_api):
    """Same contract as the terminal: no checkpoint write, no superstep, no
    model call — the question is simply asked again, under a fresh sequence,
    which is what kills the first message's buttons."""
    human = _channel(slack_api)
    question = Question(
        action="confirm_playbook", prompt="p", free_text="none", allow_skip=True
    )
    task = asyncio.create_task(human.ask(question))
    await until(lambda: human.pending_seq == 1)

    human.submit(seq=1, text="not allowed here")
    await until(lambda: human.pending_seq == 2)

    assert "does not accept free text" in slack_api.posts[1]["text"]
    assert len(slack_api.posts) == 3  # question, complaint, question again
    assert human.submit(seq=1, skip=True) == "that question has already been answered"
    task.cancel()


# --- timeout ------------------------------------------------------------------


async def test_a_timed_out_question_has_its_buttons_removed(slack_api):
    """Timing out rather than leaking the slot. Slack has no flag that disables
    a button, so the block has to go — otherwise the message sits there
    inviting a click nothing will answer."""
    human = _channel(slack_api, answer_timeout=0.01)

    with pytest.raises(QuestionTimeout):
        await human.ask(_alert_question(_C0, _C1))

    assert human.pending_seq is None
    assert len(slack_api.updates) == 1
    assert not [b for b in slack_api.updates[0]["blocks"] if b["type"] == "actions"]


# --- tell ---------------------------------------------------------------------


async def test_the_report_is_posted_into_the_thread(slack_api):
    human = _channel(slack_api)

    await human.tell(Event(type="report", text="## Root cause\n\nA GC pause."))

    assert len(slack_api.posts) == 1
    assert slack_api.posts[0]["thread_ts"] == "1700000000.000100"
    assert slack_api.posts[0]["blocks"][0]["type"] == "markdown"


async def test_a_long_report_becomes_several_messages_in_order(slack_api):
    human = _channel(slack_api)
    report = "\n\n".join(f"## Section {i}\n\n" + ("word " * 400) for i in range(12))

    await human.tell(Event(type="report", text=report))

    assert len(slack_api.posts) > 1
    assert "Section 0" in slack_api.posts[0]["blocks"][0]["text"]
    assert "Section 11" in slack_api.posts[-1]["blocks"][0]["text"]
