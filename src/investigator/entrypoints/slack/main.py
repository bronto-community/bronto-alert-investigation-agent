"""The Slack bot's process entry point."""

import argparse
import asyncio
import logging
import os
import sys
from typing import cast

from bronto_sdk import BrontoClient, BrontoConfigError
from langchain_core.language_models import BaseChatModel
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.web.async_client import AsyncWebClient

from ...graph import open_app
from ...providers import (
    SUPPORTED_PROVIDERS,
    ConfigError,
    build_model,
    validate_provider_env,
)
from ...settings import DEFAULT_MAX_MODEL_CALLS, Settings
from ...telemetry import init_telemetry
from .bolt import build_bolt_app
from .runs import MAX_LIVE_RUNS, RunRegistry
from .thread import DEFAULT_ANSWER_TIMEOUT

logger = logging.getLogger(__name__)

BOT_TOKEN = "SLACK_BOT_TOKEN"
APP_TOKEN = "SLACK_APP_TOKEN"


def main(argv: list[str] | None = None) -> None:
    """Validate configuration, then serve until interrupted."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        init_telemetry()
    try:
        client = BrontoClient.from_env()
        validate_provider_env(args.provider)
        bot_token = _require(BOT_TOKEN)
        app_token = _require(APP_TOKEN)
    except (BrontoConfigError, ConfigError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None

    settings = Settings(
        provider=args.provider,
        model_id=args.model_id,
        max_model_calls=args.max_model_calls,
        log_level=args.log_level,
    )
    # Said out loud at startup rather than discovered during an incident.
    logger.warning(
        "in-flight investigations do not survive a restart; a paused run is "
        "held in memory and abandoned after %d minutes without an answer",
        int(DEFAULT_ANSWER_TIMEOUT // 60),
    )
    try:
        asyncio.run(
            serve(
                client,
                build_model(settings.provider, settings.model_id),
                settings=settings,
                bot_token=bot_token,
                app_token=app_token,
                max_live=args.max_live,
            )
        )
    except KeyboardInterrupt:
        logger.info("shutting down")


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"{name} is not set")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="investigator-slack",
        description="Run the investigator as a Slack bot over Socket Mode.",
    )
    parser.add_argument("--provider", required=True, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--model-id", dest="model_id", required=True)
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    parser.add_argument(
        "--max-model-calls",
        dest="max_model_calls",
        type=int,
        default=DEFAULT_MAX_MODEL_CALLS,
        help="model-call budget for each investigation loop",
    )
    parser.add_argument(
        "--max-live",
        dest="max_live",
        type=int,
        default=MAX_LIVE_RUNS,
        help="concurrent investigations to allow (default: %(default)s)",
    )
    return parser.parse_args(argv)


REQUIRED_SCOPES = (
    "commands",
    "chat:write",
    "app_mentions:read",
    "channels:history",
    "groups:history",
)


def missing_scopes(granted: str | None) -> list[str]:
    """Which required scopes the `x-oauth-scopes` header does not list.

    An absent or empty header is Slack declining to say, not a token with no
    scopes, so it yields nothing to warn about.
    """
    if not granted:
        return []
    have = {scope.strip() for scope in granted.split(",")}
    return [scope for scope in REQUIRED_SCOPES if scope not in have]


async def log_identity_and_scopes(client: AsyncWebClient) -> None:
    """Say who the bot is and what it may do, before it starts listening.

    Warns and returns on any failure: a diagnostic that can take the bot down
    is worse than the bug it was added to find.
    """
    try:
        response = await client.auth_test()
        logger.info(
            "connected as %s (%s) in %s",
            response.get("user"),
            response.get("user_id"),
            response.get("team"),
        )
        headers = cast(dict[str, str], response.headers or {})
        if absent := missing_scopes(headers.get("x-oauth-scopes")):
            logger.warning(
                "the bot token is missing %s — reinstall the Slack app to grant "
                "them, or @mentions and thread replies will never arrive",
                ", ".join(absent),
            )
    except Exception:  # never let the check itself stop the bot
        logger.warning("could not check the bot token's scopes", exc_info=True)


async def serve(
    client: BrontoClient,
    model: BaseChatModel,
    *,
    settings: Settings,
    bot_token: str,
    app_token: str,
    max_live: int = MAX_LIVE_RUNS,
) -> None:
    """Open one app (see ``open_app``), then serve until stopped."""
    async with open_app(client, model, settings) as graph:
        registry = RunRegistry(graph, max_live=max_live)
        bolt = build_bolt_app(registry, token=bot_token)
        await log_identity_and_scopes(bolt.client)
        handler = AsyncSocketModeHandler(bolt, app_token)
        logger.info("connecting to Slack over Socket Mode")
        try:
            await handler.start_async()
        finally:
            await registry.shutdown()


if __name__ == "__main__":
    main()
