"""Tests for the write-only Slack sink used by the event deployment mode."""

import pytest
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

from investigator.entrypoints.slack.post import (
    SlackSink,
    WebClientSlackApi,
    build_sink,
)
from investigator.human import Event, NoOneToAsk, Question
from tests.entrypoints.slack.conftest import FakeSlackApi


@pytest.mark.parametrize("kind", ["report", "notice"])
async def test_the_sink_posts_what_it_is_told(kind):
    """A report, and a failed run's notice, go out through the same object."""
    sink = SlackSink(FakeSlackApi())

    await sink.tell(Event(type=kind, text="# Root cause\n\nThe heap filled."))

    assert len(sink._api.posts) == 1
    assert "heap filled" in str(sink._api.posts[0]["blocks"])


async def test_a_long_report_is_split_and_posted_in_order():
    """render_event splits past Slack's markdown limit, and the pieces have to
    arrive in the order they were written."""
    sink = SlackSink(FakeSlackApi())
    paragraphs = "\n\n".join(f"Paragraph {i}. " + "x" * 400 for i in range(40))

    await sink.tell(Event(type="report", text=paragraphs))

    posts = sink._api.posts
    assert len(posts) > 1
    rendered = [str(post["blocks"]) for post in posts]
    assert "Paragraph 0." in rendered[0]
    assert "Paragraph 39." in rendered[-1]


async def test_being_asked_a_question_names_the_question():
    """Unreachable by construction — an event run enters at context generation
    and nothing downstream interrupts — so if it ever fires, say which one."""
    sink = SlackSink(FakeSlackApi())

    with pytest.raises(NoOneToAsk, match="confirm_alert"):
        await sink.ask(Question(action="confirm_alert", prompt="which one?"))


async def test_the_adapter_posts_to_a_channel(mocker):
    """The regression guard: `channel` is a required keyword-only argument, and
    a bare Mock would accept any spelling of it."""
    client = mocker.create_autospec(AsyncWebClient, instance=True)
    client.chat_postMessage.return_value = {"ts": "1.0"}

    ts = await WebClientSlackApi(client, "C1").post(blocks=[], text="hi")

    assert ts == "1.0"
    assert client.chat_postMessage.call_args.kwargs["channel"] == "C1"
    assert client.chat_postMessage.call_args.kwargs["thread_ts"] is None


def test_the_sink_retries_rate_limits():
    """Unattended, a 429 partway through a multi-message report would silently
    truncate it."""
    sink = build_sink("xoxb-fake", "C1")

    handlers = sink._api._client.retry_handlers
    assert any(isinstance(h, AsyncRateLimitErrorRetryHandler) for h in handlers)
