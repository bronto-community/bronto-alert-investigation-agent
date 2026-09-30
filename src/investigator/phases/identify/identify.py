"""Phase 1: identify the alert which matches user's natural language description.

A Reason/Act agent that either searches audit events or asks for human input.

Two human-in-the-loop questions live next door, split across confirm.py and
questions.py.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

# langchain leaves create_agent's checkpointer type argument unknown.
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from ...settings import Deps
from ...state import InvestigationState
from .confirm import (
    IdentificationState,
    PlaybookLookupMiddleware,
    confirm_playbook,
    make_confirm_alert_tool,
)
from .questions import INTERRUPT_ON
from .search import (
    fetch_configured_playbook,
    make_alert_timeseries_tool,
    make_audit_events_tool,
)

logger = logging.getLogger(__name__)


def build_identification_node(deps: Deps):
    """Build the identification node: a Reason/Act agent behind one graph node."""
    tools = [
        make_alert_timeseries_tool(deps.client),
        make_audit_events_tool(deps.client),
        make_confirm_alert_tool(description=CONFIRM_DESCRIPTION),
        confirm_playbook,
    ]
    # Order matters: `after_model` hooks run in reverse, so listing the playbook
    # lookup after the human-in-the-loop middleware is what makes it run
    # *before* it — the playbook is in state by the time the question renders.
    middleware: list[AgentMiddleware[Any]] = [
        ForceToolChoiceMiddleware(),
        HumanInTheLoopMiddleware(INTERRUPT_ON),
        PlaybookLookupMiddleware(make_playbook_lookup(deps)),
    ]
    agent: CompiledStateGraph[Any] = create_agent(
        deps.model,
        tools,
        system_prompt=system_prompt(SYSTEM_PROMPT, datetime.now(UTC)),
        state_schema=IdentificationState,
        middleware=middleware,
        name="identification",
    )

    async def identification(state: InvestigationState) -> dict[str, Any]:
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(state["request_text"])]},
                {"recursion_limit": RECURSION_LIMIT},
            )
        except GraphRecursionError:
            # Searching forever without ever proposing a candidate is a failure
            # to identify the alert, so report it as one instead of crashing the
            # run — the orchestrator routes an unset triggered alert to give_up.
            logger.warning(
                "identification exhausted its loop budget without a triggered alert"
            )
            return {"triggered_alert": None, "playbook": None}
        return {
            "triggered_alert": result.get("triggered_alert"),
            "playbook": result.get("playbook"),
        }

    return identification


# LangGraph supersteps (roughly two per model turn) the agent may take before
# identification gives up. Hitting it is a failure to identify the alert,
# reported as one rather than crashing the run.
RECURSION_LIMIT = 18


def system_prompt(base: str, now: datetime) -> str:
    """The system prompt plus today's date as time-reasoning ground truth.

    Built once when the agent is constructed rather than injected per turn, so
    the cached prompt prefix stays byte-identical across a run's turns.
    """
    return (
        f"{base}\n"
        f"Today's date is {now.date().isoformat()}. Use this as ground truth for "
        "any time-relative reasoning (e.g. 'current year', 'recent', 'latest')."
    )


SYSTEM_PROMPT = (
    "You are an SRE assistant identifying which triggered alert a user is asking "
    "about. Every turn must be a tool call: either search triggered alerts with "
    "the audit tools, or call confirm_alert with the alert(s) that best match "
    "the user's description. Never answer in prose — once the search results "
    "contain a plausible match, propose it instead of describing it. When one "
    "alert is a highly likely match, propose only that single candidate; propose "
    "multiple candidates when no candidate is an obvious choice. Prefer grounded "
    "matches over guesses.\n"
    "Once the user confirms an alert, immediately call confirm_playbook, passing "
    "the alert's configured playbook verbatim if the tool results show one and "
    "null otherwise. Call it on its own, not batched with other tools."
)

CONFIRM_DESCRIPTION = (
    "Ask the user which triggered alert they are investigating.\n\n"
    "Call this once the audit searches surface one or more plausible matches, "
    "instead of continuing to search. Propose a single candidate when one is a "
    "highly likely match, and several only when no candidate is an obvious "
    "choice. The user may accept one, pick another, or send you back to search "
    "again with more detail."
)


# --- the playbook -------------------------------------------------------------


def make_playbook_lookup(deps: Deps) -> Callable[[dict[str, Any]], str | None]:
    """Fetch the playbook configured on the confirmed alert's monitor.

    The id comes off the audit row on the triggered alert, not from the model.
    """

    def lookup(triggered_alert: dict[str, Any]) -> str | None:
        data: dict[str, Any] = triggered_alert.get("data") or {}
        monitor_id = data.get("monitor_id")
        if not monitor_id:
            return None
        return fetch_configured_playbook(deps.client, monitor_id)

    return lookup


# --- forcing a tool call every turn -------------------------------------------


class ForceToolChoiceMiddleware(AgentMiddleware):
    """Require a tool call on every turn."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(tool_choice="any"))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(tool_choice="any"))
