"""Posting into Slack: the web-client adapter, and the write-only sink.

Two users share the adapter. The bolt app injects a client it already has; the
webhook's ``--report slack`` builds its own from a bot token. Neither knows
anything about the other, and both go through the same two-method ``SlackApi``
seam the rendering is written against.

``SlackSink`` is what lets an unattended webhook run report into Slack: the run
still ends by handing its report to a ``Human``, so posting it to a channel is a
``Human`` that can only speak, never listen.
"""

import logging
from typing import Any, cast

from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

from ...human import Answer, Event, NoOneToAsk, Question
from .blocks import render_event

logger = logging.getLogger(__name__)


class WebClientSlackApi:
    """Adapts an AsyncWebClient to the two calls a run makes."""

    def __init__(self, client: AsyncWebClient, channel_id: str) -> None:
        self._client = client
        self._channel_id = channel_id

    async def post(
        self, *, blocks: list[dict[str, Any]], text: str, thread_ts: str | None = None
    ) -> str:
        response = await self._client.chat_postMessage(
            channel=self._channel_id, thread_ts=thread_ts, blocks=blocks, text=text
        )
        ts = cast(str | None, response["ts"])
        assert ts is not None  # Slack always returns a ts on a successful post
        return ts

    async def update(self, *, ts: str, blocks: list[dict[str, Any]], text: str) -> None:
        await self._client.chat_update(
            channel=self._channel_id, ts=ts, blocks=blocks, text=text
        )


class SlackSink:
    """The ``Human`` seat of an unattended run: it posts and never asks."""

    def __init__(self, api: WebClientSlackApi) -> None:
        self._api = api

    async def ask(self, question: Question) -> Answer:
        raise NoOneToAsk(question.action)

    async def tell(self, event: Event) -> None:
        # Sequentially, not gathered: chat.postMessage is rate limited to about
        # one per second per channel, and a long report is several messages.
        for blocks, text in render_event(event):
            await self._api.post(blocks=blocks, text=text)


def build_sink(token: str, channel_id: str) -> SlackSink:
    """A sink posting to ``channel_id`` as the bot ``token`` belongs to.

    The retry handler is the point: this path is unattended, so a 429 partway
    through a multi-message report would silently truncate it.
    """
    client = AsyncWebClient(token=token)
    client.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=3))
    return SlackSink(WebClientSlackApi(client, channel_id))
