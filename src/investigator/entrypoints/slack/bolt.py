"""The bolt handlers.

Each listener is one line of delegation to a ``handle_*`` free function, which
is what the tests call — no Bolt dispatcher in the suite at all, and the
`ack()`-first rule is visible in one place.
"""

import logging
import re
from typing import Any, cast

from slack_bolt.async_app import AsyncAck, AsyncApp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from .blocks import parse_action_id
from .post import WebClientSlackApi
from .runs import RunRegistry

logger = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(r"^ai:")
_MENTION = re.compile(r"<@[UW][A-Z0-9]+>")


def strip_mention(text: str) -> str:
    """Drop the `<@U123>` the platform puts in an app_mention's text."""
    return _MENTION.sub("", text or "").strip()


# --- handlers -----------------------------------------------------------------


async def handle_command(
    registry: RunRegistry, client: AsyncWebClient, body: dict[str, Any]
) -> None:
    """Start a run from `/investigate`.

    A slash command cannot be invoked inside a thread — only built-in commands
    can — so its payload carries no thread_ts and this always mints a fresh
    root message whose ts becomes the run's thread. No ambiguity to resolve.
    """
    channel_id, user_id = body["channel_id"], body["user_id"]
    text = (body.get("text") or "").strip()
    if (refusal := registry.can_start()) is not None:
        await _ephemeral(client, channel_id, user_id, refusal)
        return
    described = text or "(no description given)"
    try:
        root = await client.chat_postMessage(
            channel=channel_id, text=f":mag: Investigating: {described}"
        )
    except SlackApiError as error:
        if error.response.get("error") == "not_in_channel":
            await _ephemeral(
                client,
                channel_id,
                user_id,
                "Invite me to this channel first, then try again.",
            )
            return
        raise
    registry.start(
        WebClientSlackApi(client, channel_id),
        team_id=body.get("team_id", "T"),
        channel_id=channel_id,
        thread_ts=cast(str, root["ts"]),
        request_text=text,
    )


async def handle_mention(
    registry: RunRegistry, client: AsyncWebClient, event: dict[str, Any]
) -> None:
    """Start a run from an @mention, in the thread it happened in.

    An @mention inside a live thread arrives TWICE — once here and once as a
    `message` event. Returning early when the thread already has a run is what
    stops us both answering the pending question and starting a second,
    competing run in the same place.
    """
    channel_id = event["channel"]
    thread_ts = event.get("thread_ts") or event["ts"]
    logger.info("mention in %s thread %s", channel_id, thread_ts)
    if registry.get(channel_id, thread_ts) is not None:
        logger.info("thread %s already has a run; ignoring the mention", thread_ts)
        return
    if (refusal := registry.can_start()) is not None:
        await _ephemeral(client, channel_id, event.get("user", ""), refusal)
        return
    described = strip_mention(event.get("text", ""))
    try:
        await client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=f":mag: Investigating: {described or '(no description given)'}",
        )
    except SlackApiError:
        # The acknowledgement is a courtesy; the investigation is the point.
        logger.warning("could not acknowledge the mention", exc_info=True)
    registry.start(
        WebClientSlackApi(client, channel_id),
        team_id=event.get("team", "T"),
        channel_id=channel_id,
        thread_ts=thread_ts,
        request_text=described,
    )


async def handle_action(
    registry: RunRegistry, client: AsyncWebClient, body: dict[str, Any]
) -> None:
    """Apply a button or select choice to the question it was posted for."""
    action = body["actions"][0]
    parsed = parse_action_id(action.get("action_id", ""))
    if parsed is None:
        return
    seq, control, option_id = parsed
    if control == "select":
        # A select carries the answer in the chosen option's value, since one
        # action_id serves every option.
        raw = action["selected_option"]["value"]
        seq_text, _, option_id = raw.partition(":")
        seq = int(seq_text)

    channel_id, user_id = body["channel"]["id"], body["user"]["id"]
    message: dict[str, Any] = body.get("message") or {}
    thread_ts = message.get("thread_ts") or message.get("ts")
    if thread_ts is None:
        # A payload with no message to trace back to a thread cannot belong to
        # any run of ours. Slack always sends one; a malformed retry might not.
        return
    handle = registry.get(channel_id, thread_ts)
    if handle is None:
        await _ephemeral(
            client, channel_id, user_id, "That investigation is no longer running."
        )
        return

    refusal = handle.human.submit(
        seq=seq,
        option_id=None if control == "skip" else option_id,
        skip=control == "skip",
    )
    if refusal:
        # Ephemeral, not a thread reply: a stale click is the clicker's
        # problem and should not clutter everyone else's incident thread.
        await _ephemeral(client, channel_id, user_id, refusal)


def handle_message(registry: RunRegistry, event: dict[str, Any]) -> None:
    """Treat a thread reply as an answer, when it actually is one.

    This fires for every message in every channel the bot is in, so the guards
    matter more than the body. Synchronous on purpose: `submit` only sets a
    Future result, so a reply can never be lost at an await point.
    """
    if event.get("subtype") is not None:
        return  # joins, edits, deletes, file shares, bot_message
    if event.get("bot_id") or event.get("app_id"):
        # Bolt's AsyncIgnoringSelfEvents already drops our own messages; this
        # covers other bots and webhooks posting into the thread.
        return
    thread_ts = event.get("thread_ts")
    if thread_ts is None:
        return  # top-level chatter is never an answer
    handle = registry.get(event.get("channel", ""), thread_ts)
    if handle is None:
        return  # not a thread we started
    human = handle.human
    seq = human.pending_seq
    if seq is None or not human.accepts_text:
        # No pending question, or a buttons-only one. Someone saying "thanks"
        # should not be punished with the question re-posted at them.
        return
    logger.info("thread reply answering question %d in %s", seq, thread_ts)
    human.submit(seq=seq, text=event.get("text", ""))


async def _ephemeral(
    client: AsyncWebClient, channel_id: str, user_id: str, text: str
) -> None:
    try:
        await client.chat_postEphemeral(channel=channel_id, user=user_id, text=text)
    except SlackApiError:
        logger.warning("could not post an ephemeral notice", exc_info=True)


# --- wiring -------------------------------------------------------------------


def build_bolt_app(registry: RunRegistry, *, token: str) -> AsyncApp:
    """Wire the handlers onto a bolt app.

    Every listener acks first and delegates second. Slack times out at three
    seconds and Socket Mode redelivers, so an await before the ack is how you
    get an investigation started twice.
    """
    app = AsyncApp(token=token)  # Socket Mode needs no signing secret

    @app.command("/investigate")
    async def _command(
        ack: AsyncAck, body: dict[str, Any], client: AsyncWebClient
    ) -> None:
        await ack()
        await handle_command(registry, client, body)

    @app.event("app_mention")
    async def _mention(event: dict[str, Any], client: AsyncWebClient) -> None:
        await handle_mention(registry, client, event)

    @app.action(ACTION_PATTERN)
    async def _action(
        ack: AsyncAck, body: dict[str, Any], client: AsyncWebClient
    ) -> None:
        await ack()
        await handle_action(registry, client, body)

    @app.event("message")
    async def _message(event: dict[str, Any]) -> None:
        handle_message(registry, event)

    @app.error
    async def _error(error: Exception, body: dict[str, Any]) -> None:
        logger.exception("unhandled slack error: %s", error)

    return app
