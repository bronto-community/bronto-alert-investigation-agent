"""The receiver: a Bronto webhook in, an investigation out.

The one place this project accepts inbound HTTP, and the whole of what it knows
about the web. It authenticates the caller, validates the payload with the same
function the CLI uses, drops retries, and hands the alert to ``EventRuns``.
"""

import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from hmac import compare_digest
from typing import Any, cast

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from ...state import state_for_event
from .runs import AtCapacity, alert_key

logger = logging.getLogger(__name__)

# What Bronto must send. Confirm this against the webhook's own configuration
# before deploying: a header name mismatch looks exactly like a wrong secret.
SECRET_HEADER = "x-bronto-webhook-secret"

# A webhook body is a monitor alert, not an upload.
MAX_BODY_BYTES = 64 * 1024

# Retries arrive within minutes, so remembering the last few hundred alerts is
# enough to swallow them. Deliberately in memory and deliberately small: a
# restart forgetting them costs one duplicate investigation.
RECENT_EVENTS = 512

# Takes the validated payload, returns the run's id, raises ``AtCapacity``.
# Synchronous on purpose: the slot has to be taken before this handler awaits
# anything, or two webhooks arriving together both see the last one free.
Start = Callable[[dict[str, Any]], str]


def build_receiver(start: Start, *, secret: str) -> Starlette:
    """Wire the two routes onto an app. ``start`` is the only thing it can do."""
    seen: OrderedDict[str, None] = OrderedDict()

    async def receive(request: Request) -> JSONResponse:
        if not _authentic(request, secret):
            # Before the body is read: an unauthenticated caller learns nothing
            # about what we would have parsed.
            logger.warning("refused a webhook with a bad or missing secret")
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        body = await request.body()
        try:
            payload = _parse(body)
        except ValueError as error:
            logger.warning("refused an unusable webhook payload: %s", error)
            return JSONResponse({"error": str(error)}, status_code=400)

        key = alert_key(payload)
        if key in seen:
            logger.info("ignored a retry of %s", key)
            return JSONResponse({"duplicate": True, "alert": key}, status_code=200)

        try:
            run_id = start(payload)
        except AtCapacity as refusal:
            # Not remembered, so the retry — the one thing that can still save
            # this alert — is answered properly rather than as a duplicate.
            logger.warning("refused %s: %s", key, refusal)
            return JSONResponse({"error": str(refusal)}, status_code=503)

        _remember(seen, key)
        logger.info("started %s for %s", run_id, key)
        return JSONResponse({"run_id": run_id, "alert": key}, status_code=202)

    async def health(_: Request) -> PlainTextResponse:
        return PlainTextResponse("ok")

    return Starlette(
        routes=[
            Route("/events", receive, methods=["POST"]),
            Route("/health", health, methods=["GET"]),
        ]
    )


def _authentic(request: Request, secret: str) -> bool:
    """Constant-time, so the URL cannot be turned into an oracle for the secret."""
    return compare_digest(request.headers.get(SECRET_HEADER, ""), secret)


def _parse(body: bytes) -> dict[str, Any]:
    """The payload, or a ValueError naming what is wrong with it."""
    if len(body) > MAX_BODY_BYTES:
        raise ValueError(f"payload is larger than {MAX_BODY_BYTES} bytes")
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError(f"payload is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a JSON object, got {type(payload).__name__}")
    payload = cast(dict[str, Any], payload)
    try:
        state_for_event(payload)
    except ValidationError as error:
        raise ValueError(f"unusable event payload: {error}") from error
    return payload


def _remember(seen: "OrderedDict[str, None]", key: str) -> None:
    seen[key] = None
    while len(seen) > RECENT_EVENTS:
        seen.popitem(last=False)
