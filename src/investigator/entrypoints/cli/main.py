"""The command-line adapter."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, cast

from bronto_sdk import BrontoClient, BrontoConfigError
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from ...graph import open_app
from ...human import Human
from ...providers import (
    SUPPORTED_PROVIDERS,
    ConfigError,
    build_model,
    validate_provider_env,
)
from ...run import DESCRIBE_REQUEST, run_config, run_to_completion
from ...settings import DEFAULT_MAX_MODEL_CALLS, Settings
from ...state import InvestigationState, state_for_event, state_for_request
from ...telemetry import init_telemetry
from .terminal import TerminalHuman

logger = logging.getLogger(__name__)

THREAD_ID = "cli"


def main(argv: list[str] | None = None) -> None:
    """Parse args, validate configuration, and start the investigation."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s"
    )
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        init_telemetry()
    try:
        client = BrontoClient.from_env()
        validate_provider_env(args.provider)
    except (BrontoConfigError, ConfigError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from None
    event = _load_event(args.event) if args.event else None
    settings = Settings(
        provider=args.provider,
        model_id=args.model_id,
        max_model_calls=args.max_model_calls,
        log_level=args.log_level,
    )
    logger.info(
        "investigating with provider=%s model=%s", settings.provider, settings.model_id
    )
    asyncio.run(
        run(
            client,
            build_model(settings.provider, settings.model_id),
            settings=settings,
            event=event,
        )
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="investigator")
    parser.add_argument("--provider", required=True, choices=SUPPORTED_PROVIDERS)
    parser.add_argument("--model-id", dest="model_id", required=True)
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    parser.add_argument(
        "--max-model-calls",
        dest="max_model_calls",
        type=int,
        default=DEFAULT_MAX_MODEL_CALLS,
        help="model-call budget for the investigation loop",
    )
    parser.add_argument(
        "--event",
        dest="event",
        help="investigate an already-identified event from this JSON file "
        "('-' for stdin), skipping identification entirely",
    )
    return parser.parse_args(argv)


def _load_event(path: str) -> dict[str, Any]:
    """Read and validate an event payload from a file, or from stdin for '-'.

    Validated here rather than inside the run: a malformed webhook should exit
    non-zero in a second — which is what a queue worker branches on — not after
    an MCP session and a model call.
    """
    try:
        raw = sys.stdin.read() if path == "-" else Path(path).read_text()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"event payload must be a JSON object, got {type(payload).__name__}"
            )
    except (OSError, ValueError) as error:
        print(f"could not read --event: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    event = cast(dict[str, Any], payload)
    try:
        state_for_event(event)
    except ValidationError as error:
        print(f"unusable --event payload: {error}", file=sys.stderr)
        raise SystemExit(1) from None
    return event


async def run(
    client: BrontoClient,
    model: BaseChatModel,
    *,
    settings: Settings,
    event: dict[str, Any] | None,
) -> None:
    """Open the app (see ``open_app``) for the duration of one investigation."""
    async with open_app(client, model, settings) as app:
        await investigate(app, TerminalHuman(), event)


async def investigate(
    app: CompiledStateGraph[InvestigationState],
    human: Human,
    event: dict[str, Any] | None,
) -> None:
    """Build the run's starting state, then hand it to the shared loop."""
    if event is not None:
        # Already grounded: the graph enters at context generation, so there is
        # nothing to describe and nothing to confirm.
        initial = state_for_event(event)
        triggered_alert = initial["triggered_alert"]
        assert triggered_alert is not None  # state_for_event always grounds one
        logger.info("investigating monitor event %s", triggered_alert["id"])
    else:
        answer = await human.ask(DESCRIBE_REQUEST)
        initial = state_for_request(answer.text or "")
    await run_to_completion(
        app,
        initial,
        human,
        config=run_config(THREAD_ID),
    )
