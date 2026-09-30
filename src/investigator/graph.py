"""The graph: the four phases of an investigation, in the order they run.

Each phase is a node, and each one reads some fields off
the ``InvestigationState`` and writes others back. Nothing else passes between
them.

    1. identification      identify which triggered monitor is meant
       reads   request_text
       writes  triggered_alert, playbook
       model   unbounded calls, but a bounded loop

    2. context             gather what is known about it
       reads   triggered_alert, playbook
       writes  alert_context
       model   at most one call, and only when a playbook was supplied

    3. investigation       find the root cause
       reads   alert_context
       writes  findings
       model   --max-model-calls, default 5, the last one spent on the
               conclusion rather than another search

    4. report_generation   write it up for a human
       reads   alert_context, findings
       writes  report
       model   exactly one call, or none when there are no findings

A run that starts from a triggered-alert event (from the webhook, say) rather
than a description skips phase 1 and enters at phase 2. A run whose alert
cannot be identified routes to ``give_up``, which writes a report saying so.
"""

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from bronto_sdk import BrontoClient
from bronto_sdk import mcp as bronto_mcp
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.sessions import create_session
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .phases.context import build_context_node
from .phases.identify.identify import build_identification_node
from .phases.investigate import (
    Findings,
    build_investigation_node,
    investigation_tools,
)
from .phases.report import make_report_node
from .settings import Deps, Settings
from .state import InvestigationState

logger = logging.getLogger(__name__)


def build_app(
    deps: Deps,
    *,
    max_model_calls: int,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[InvestigationState]:
    """Compile the orchestrator."""
    builder = StateGraph(InvestigationState)

    builder.add_node("identification", build_identification_node(deps))
    builder.add_node("context", build_context_node(deps))
    builder.add_node("give_up", give_up)
    builder.add_node("investigation", build_investigation_node(deps, max_model_calls))
    builder.add_node("report_generation", make_report_node(deps.model))

    builder.add_conditional_edges(START, route_entry, ["identification", "context"])
    builder.add_conditional_edges(
        "identification", route_after_identification, ["context", "give_up"]
    )
    builder.add_edge("context", "investigation")
    builder.add_edge("give_up", END)
    builder.add_edge("investigation", "report_generation")
    builder.add_edge("report_generation", END)

    return builder.compile(checkpointer=checkpointer)


def route_entry(state: InvestigationState) -> str:
    """Where a run starts: identification, or straight into context generation."""
    decision = "context" if state.get("triggered_alert") else "identification"
    logger.info("route at entry -> %s", decision)
    return decision


def route_after_identification(state: InvestigationState) -> str:
    """Continue only once a triggered alert has actually been confirmed."""
    decision = "context" if state.get("triggered_alert") else "give_up"
    logger.info("route after identification -> %s", decision)
    return decision


def give_up(state: InvestigationState) -> dict[str, Any]:
    """The one terminal node for a run that never identified a triggered alert."""
    logger.warning("giving up: no monitor identified")
    return {"report": GIVE_UP_MESSAGE}


GIVE_UP_MESSAGE = (
    "I'm struggling to find the alert related to your description. Please "
    "try again with more detail, such as the alert name, the approximate "
    "time, or the affected service."
)


@asynccontextmanager
async def open_app(
    client: BrontoClient, model: BaseChatModel, settings: Settings
) -> AsyncGenerator[CompiledStateGraph[InvestigationState], None]:
    """Open the MCP session and yield an app compiled around it.

    The key and region come from the same environment variables
    ``BrontoClient.from_env()`` has already validated at startup.
    ``BRONTO_MCP_BASE_URL`` overrides the region's host, for staging. It is not
    ``BRONTO_BASE_URL``: that one names the REST host, a different origin.
    """
    url = bronto_mcp.url(
        os.environ.get("BRONTO_REGION"), base_url=os.environ.get("BRONTO_MCP_BASE_URL")
    )
    headers = bronto_mcp.auth_headers(api_key=os.environ.get("BRONTO_API_KEY"))
    logger.info("opening MCP session at %s", url)
    async with create_session(
        {"transport": "streamable_http", "url": url, "headers": headers}
    ) as session:
        # create_session connects but leaves the handshake to the caller.
        await session.initialize()
        tools = await load_mcp_tools(session)
        logger.info("loaded %d MCP tool(s)", len(tools))
        deps = Deps(
            client=client, model=model, investigation_tools=investigation_tools(tools)
        )
        yield build_app(
            deps,
            max_model_calls=settings.max_model_calls,
            checkpointer=build_checkpointer(),
        )


def build_checkpointer() -> MemorySaver:
    """An in-memory checkpointer that knows the one non-JSON value in a run."""
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=(Findings,)))
