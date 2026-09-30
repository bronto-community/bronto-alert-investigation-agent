"""Identification phase's two human-in-the-loop tools.

Both questions the agent asks — which candidate the user means, and whether
to supply a playbook — are modelled as tools rather than as interrupt nodes,
because ``HumanInTheLoopMiddleware`` only ever interrupts on a tool call.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from ...state import TriggeredAlert
from .search import find_audit_row

logger = logging.getLogger(__name__)

CONFIRM_ALERT = "confirm_alert"
CONFIRM_PLAYBOOK = "confirm_playbook"


# --- the two tools ------------------------------------------------------------


class IdentificationState(AgentState):
    """The identification agent's state.

    ``triggered_alert`` is the confirmed alert as a grounded ``TriggeredAlert`` and
    ``playbook`` the user's optional extra context — the two things this phase
    hands the rest of the pipeline. ``configured_playbook`` is scratch: the
    monitor's pre-configured playbook, fetched just in time to show the user.
    """

    triggered_alert: NotRequired[dict[str, Any] | None]
    configured_playbook: NotRequired[str | None]
    playbook: NotRequired[str | None]


class Candidate(BaseModel):
    """One triggered alert the agent proposes, copied from an audit search row."""

    id: str = Field(description="The monitor_id of an audit search result.")
    label: str = Field(description="Its monitor_name.")
    occurred_at: str | None = Field(
        default=None, description="Its triggered_at, copied verbatim."
    )


def make_confirm_alert_tool(*, description: str) -> BaseTool:
    """Build the confirm-the-candidate tool."""

    @tool(CONFIRM_ALERT, description=description)
    def confirm_alert(
        candidates: list[Candidate], runtime: ToolRuntime[None, IdentificationState]
    ) -> Command[Any]:
        if len(candidates) != 1:
            # The interrupt narrows a multi-candidate proposal to the user's
            # pick before the body runs, so more than one here means the
            # selection was bypassed rather than made.
            raise ValueError(
                f"{CONFIRM_ALERT} must be narrowed to one candidate before "
                f"execution, got {len(candidates)}"
            )
        candidate = candidates[0]
        row = find_audit_row(
            runtime.state["messages"], candidate.id, candidate.occurred_at
        )
        if row is None:
            logger.error(
                "confirmed candidate %s: not in any retrieved artifact", candidate.id
            )
            raise ValueError(
                f"confirmed candidate {candidate.id} not found in retrieved artifacts"
            )
        logger.info("alert confirmed: %s", candidate.id)
        triggered_alert = TriggeredAlert(
            id=candidate.id,
            label=candidate.label,
            occurred_at=candidate.occurred_at,
            data=row,
        )
        return Command(
            update={
                "triggered_alert": triggered_alert.model_dump(mode="json"),
                "messages": [
                    ToolMessage(content="confirmed", tool_call_id=runtime.tool_call_id)
                ],
            }
        )

    return confirm_alert


@tool(return_direct=True)
def confirm_playbook(
    playbook: str | None, runtime: ToolRuntime[None, IdentificationState]
) -> Command[Any]:
    """Ask the user for an optional playbook.

    Call this immediately after the user confirms an alert, and call it on its
    own rather than batched with a search. Pass the alert's configured playbook
    verbatim if the tool results showed one, otherwise null — the user can
    reuse it, replace it, or skip.
    """
    logger.info("playbook confirmed (provided=%s)", playbook is not None)
    return Command(
        update={
            "playbook": playbook,
            "messages": [
                ToolMessage(
                    content="playbook recorded", tool_call_id=runtime.tool_call_id
                )
            ],
        }
    )


def _last_ai_message(messages: list[AnyMessage]) -> AIMessage | None:
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


class PlaybookLookupMiddleware(AgentMiddleware[IdentificationState]):
    """Fetch the monitor's configured playbook just before the user is asked.

    ``after_model`` hooks run in reverse middleware order, so listing this after
    the human-in-the-loop middleware makes it run *before* it — the lookup is
    committed to state by the time the description callable reads it. That keeps
    the playbook step at one model call, with no extra turn spent asking the
    model to fetch it, and no blocking REST call inside the render callback.

    ``lookup`` receives the confirmed alert, so the id it uses comes from a
    retrieved artifact rather than a model-supplied argument — there is nothing
    here to hallucinate. With no configured playbook the lookup passes None
    and the middleware is a no-op.
    """

    state_schema = IdentificationState

    def __init__(
        self, lookup: Callable[[dict[str, Any]], str | None] | None = None
    ) -> None:
        super().__init__()
        self._lookup = lookup

    def _asked(self, state: IdentificationState) -> bool:
        last = _last_ai_message(state["messages"])
        return bool(
            last
            and any(
                call["name"] == CONFIRM_PLAYBOOK for call in (last.tool_calls or [])
            )
        )

    def after_model(
        self, state: IdentificationState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not self._asked(state):
            return None
        triggered_alert = state.get("triggered_alert")
        if self._lookup is None or not triggered_alert:
            return {"configured_playbook": None}
        return {"configured_playbook": self._lookup(triggered_alert)}

    async def aafter_model(
        self, state: IdentificationState, runtime: Runtime
    ) -> dict[str, Any] | None:
        if not self._asked(state):
            return None
        triggered_alert = state.get("triggered_alert")
        if self._lookup is None or not triggered_alert:
            return {"configured_playbook": None}
        # The lookup may block (it is a `requests` call), so it
        # is offloaded to keep the MCP session's reader running on this loop.
        return {
            "configured_playbook": await asyncio.to_thread(
                self._lookup, triggered_alert
            )
        }
