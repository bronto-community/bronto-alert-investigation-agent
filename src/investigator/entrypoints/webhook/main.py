"""The webhook surface's process entry point."""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable

from bronto_sdk import BrontoClient, BrontoConfigError
from langchain_core.language_models import BaseChatModel

from ...graph import open_app
from ...human import Human, LogSink
from ...providers import (
    SUPPORTED_PROVIDERS,
    ConfigError,
    build_model,
    validate_provider_env,
)
from ...settings import DEFAULT_MAX_MODEL_CALLS, Settings
from ...telemetry import init_telemetry
from .receive import build_receiver
from .runs import MAX_LIVE_RUNS, EventRuns

logger = logging.getLogger(__name__)

WEBHOOK_SECRET = "BRONTO_WEBHOOK_SECRET"
BOT_TOKEN = "SLACK_BOT_TOKEN"


def main(argv: list[str] | None = None) -> None:
    """Validate configuration, then receive until interrupted."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        init_telemetry()
    try:
        client = BrontoClient.from_env()
        validate_provider_env(args.provider)
        secret = _require(WEBHOOK_SECRET)
        # Built once here purely to fail fast: a missing bot token should say so
        # at startup, not when the first report is ready to post.
        report_sink(args.report, args.slack_channel)
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
        "an investigation in flight does not survive a restart, and retries are "
        "de-duplicated in memory; a restart may re-investigate an alert that "
        "arrives across it"
    )
    try:
        asyncio.run(
            serve(
                client,
                build_model(settings.provider, settings.model_id),
                settings=settings,
                secret=secret,
                human_for=lambda: report_sink(args.report, args.slack_channel),
                host=args.host,
                port=args.port,
                max_live=args.max_live,
            )
        )
    except KeyboardInterrupt:
        logger.info("shutting down")


def report_sink(report: str, slack_channel: str | None) -> Human:
    """Where one run's report goes. Called per run, so no sink is shared."""
    if report != "slack":
        return LogSink()
    if not os.environ.get(BOT_TOKEN):
        raise ConfigError(f"{BOT_TOKEN} is not set (needed by --report slack)")
    try:
        from ..slack.post import build_sink
    except ImportError as error:
        raise ConfigError(
            "--report slack needs the slack extra: uv sync --extra slack"
        ) from error
    assert slack_channel is not None  # argparse guarantees it
    return build_sink(os.environ[BOT_TOKEN], slack_channel)


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"{name} is not set")
    return value


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="investigator-webhook",
        description="Investigate alerts posted to an HTTP endpoint.",
    )
    parser.add_argument("--provider", required=True, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--model-id", dest="model_id", required=True)
    parser.add_argument(
        "--report",
        choices=("logs", "slack"),
        default="logs",
        help="where each report is written (default: %(default)s)",
    )
    parser.add_argument(
        "--slack-channel",
        dest="slack_channel",
        help="channel id reports are posted to (required by --report slack)",
    )
    parser.add_argument("--host", default="0.0.0.0")  # the host's ingress
    parser.add_argument("--port", type=int, default=8080)
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
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    args = parser.parse_args(argv)
    if args.report == "slack" and not args.slack_channel:
        # The CLI makes the same demand for the same reason: an unattended run
        # has nowhere to put a report unless it is told where.
        parser.error("--report slack requires --slack-channel")
    return args


async def serve(
    client: BrontoClient,
    model: BaseChatModel,
    *,
    settings: Settings,
    secret: str,
    human_for: Callable[[], Human],
    host: str = "0.0.0.0",  # the host's ingress
    port: int = 8080,
    max_live: int = MAX_LIVE_RUNS,
) -> None:
    """Open one app (see ``open_app``), then receive until stopped."""
    import uvicorn  # the webhook extra, not a core import

    async with open_app(client, model, settings) as graph:
        runs = EventRuns(graph, human_for=human_for, max_live=max_live)
        config = uvicorn.Config(
            build_receiver(runs.start, secret=secret),
            host=host,
            port=port,
            log_level=settings.log_level.lower(),
        )
        logger.info("receiving alerts on %s:%d", host, port)
        try:
            await uvicorn.Server(config).serve()
        finally:
            await runs.shutdown()


if __name__ == "__main__":
    main()
