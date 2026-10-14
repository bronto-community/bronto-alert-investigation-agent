"""Phase 3: investigating a triggered alert.

A Reason/Act agent over the Bronto MCP tools, bounded so it always concludes.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any, NotRequired

# langchain leaves create_agent's checkpointer type argument unknown.
from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    ToolRetryMiddleware,
)
from langchain.agents.structured_output import (
    ProviderStrategy,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from ..providers import can_force_tool_choice
from ..settings import DEFAULT_MAX_MODEL_CALLS, Deps
from ..state import InvestigationState, alert_context_text

logger = logging.getLogger(__name__)


def build_investigation_node(
    deps: Deps, max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
):
    """Build the investigation node: a Reason/Act agent behind one graph node.

    The agent keeps its transcript in its own checkpoint namespace, off the
    shared state. The budget is enforced by spending the last turn on the
    conclusion rather than by cutting the loop off, so the phase ends with
    findings.
    """
    # A model that rejects forced tool use answers in the provider's native schema
    # instead, on every turn including the last.
    if can_force_tool_choice(deps.model):
        response_format, conclusion = Findings, ToolStrategy(Findings)
    else:
        response_format = conclusion = ProviderStrategy(Findings)
    # Typed wide on purpose: create_agent's input type knows only `messages`,
    # not the `alert_context` channel DiagnosisState adds.
    agent: CompiledStateGraph[Any] = create_agent(
        deps.model,
        deps.investigation_tools,
        system_prompt=SYSTEM_PROMPT,
        state_schema=DiagnosisState,
        response_format=response_format,
        middleware=[
            InvestigationWindowMiddleware(),
            ToolRetryMiddleware(max_retries=2, on_failure="continue"),
            ConcludeWithinBudgetMiddleware(max_model_calls, conclusion),
        ],
        name="investigation",
    )

    async def investigation(state: InvestigationState) -> dict[str, Any]:
        alert_context = state["alert_context"]
        try:
            result = await agent.ainvoke(
                {
                    "messages": [HumanMessage(alert_context_text(alert_context))],
                    # Must match the channel DiagnosisState declares. A key that
                    # matches none is dropped without a warning, and the window
                    # middleware then silently scopes nothing.
                    "alert_context": alert_context,
                }
            )
        except StructuredOutputValidationError:
            logger.warning(
                "investigation returned findings that did not parse", exc_info=True
            )
            return {"findings": None}
        findings = result.get("structured_response")
        if findings is None:
            logger.warning("investigation produced no findings")
            return {"findings": None}
        logger.info("investigator concluded the investigation")
        return {"findings": findings.model_dump()}

    return investigation


class DiagnosisState(AgentState):
    """The investigation agent's state: a transcript plus the alert context.

    ``alert_context`` is passed in by the orchestrator and read by the window
    middleware, so it is checkpointed with the rest of the agent's state.
    """

    alert_context: NotRequired[dict[str, Any] | None]


SYSTEM_PROMPT = (
    "You are an SRE investigating a reported problem. You are given an "
    "alert context (the monitor, the triggered event, the datasets it "
    "applies to, related datasets, and the monitor's state leading up to the "
    "trigger). Use the tools to gather evidence, identify the root cause, and "
    "suggest remediation. Prefer issuing several tool calls at once so each "
    "reasoning turn covers as much ground as possible. Searches are already "
    "scoped to the relevant historical time window — do not set time ranges "
    "(they default to that window). Keep searching until the evidence explains "
    "it, then give your findings."
)


class Findings(BaseModel):
    """The investigation's final findings.

    Give these instead of another search once the evidence explains it, or
    when asked to conclude. Give the root cause, the supporting evidence
    (grounded in what the tools actually returned), concrete remediation / next
    steps, and a confidence level. If the evidence is inconclusive, give your
    best-effort analysis and say what remains uncertain.
    """

    root_cause: str = Field(description="The single most likely cause.")
    evidence: list[str] = Field(
        description="Specific observations from tool results that support the root "
        "cause. Quote real values — log lines, counts, timestamps — not summaries."
    )
    remediation: list[str] = Field(
        description="Concrete next steps or fixes for the on-call engineer."
    )
    confidence: str = Field(
        description="How well the evidence supports the root cause, and what "
        "remains uncertain."
    )


# The strict Bronto MCP tool subset the investigator may use, in bind order.
INVESTIGATION_TOOL_NAMES = ("get_keys", "get_key_values", "search_logs", "timeseries")

# Why `time_range` never reaches a tool. The MCP search tools accept a relative
# `time_range` such as "Last 20 minutes", which the server resolves against the
# current time. An investigation looks at an alert that has already fired, so
# "the last 20 minutes" would search the wrong period. Two pieces below keep it
# out: `investigation_tools` removes the field from the schemas the model sees,
# and `InvestigationWindowMiddleware` drops it from any call that sends it anyway,
# filling in the alert's absolute epoch-ms window instead.


def investigation_tools(tools: list[BaseTool]) -> list[BaseTool]:
    """Pick the investigation subset from the loaded MCP tools, time_range param hidden.

    Raises if a required tool is absent from the loaded set.

    Each tool is copied with only its schema replaced, which keeps the loaded
    MCP tool's coroutine, name, description and ``response_format`` intact.
    """
    by_name = {tool.name: tool for tool in tools}
    missing = [name for name in INVESTIGATION_TOOL_NAMES if name not in by_name]
    if missing:
        raise ValueError(
            f"missing required investigation tool(s): {', '.join(missing)}"
        )
    narrowed: list[BaseTool] = []
    for name in INVESTIGATION_TOOL_NAMES:
        tool = by_name[name]
        # The adapter hands over the server's inputSchema, always a JSON-schema dict.
        assert isinstance(tool.args_schema, dict)
        schema = schema_without_time_range(tool.args_schema)
        narrowed.append(tool.model_copy(update={"args_schema": schema}))
    return narrowed


def schema_without_time_range(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an MCP tool's JSON schema without the ``time_range`` field."""
    properties = schema.get("properties", {})
    if "time_range" not in properties:
        return schema
    stripped = {
        **schema,
        "properties": {k: v for k, v in properties.items() if k != "time_range"},
    }
    if isinstance(stripped.get("required"), list):
        stripped["required"] = [r for r in stripped["required"] if r != "time_range"]
    return stripped


class InvestigationWindowMiddleware(AgentMiddleware):
    """Scope every time-aware tool call to the alert context's window."""

    state_schema = DiagnosisState

    def _scoped(self, request: ToolCallRequest) -> ToolCallRequest:
        alert_context: dict[str, Any] = request.state.get("alert_context") or {}
        call = request.tool_call
        args = scope_time(call["name"], call["args"], alert_context.get("window"))
        return request.override(tool_call={**call, "args": args})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(self._scoped(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await handler(self._scoped(request))


def scope_time(
    name: str, args: dict[str, Any], window: dict[str, Any] | None
) -> dict[str, Any]:
    """Scope a time-aware tool call's args to the alert context's window.

    Drops ``time_range`` (now-anchored, and wrong for a historical window) and
    defaults ``timerange_start``/``timerange_end`` from the investigation window
    when the model omitted them — an explicit range from the model is honored.
    Args for a non-time-aware tool are returned unchanged.
    """
    if name not in TIME_AWARE_TOOLS:
        return args
    scoped = {key: value for key, value in args.items() if key != "time_range"}
    if window:
        scoped.setdefault("timerange_start", window["start_ms"])
        scoped.setdefault("timerange_end", window["end_ms"])
    return scoped


# The tools that take a time range, and so get the alert's window.
TIME_AWARE_TOOLS = ("search_logs", "timeseries")


class ConcludeWithinBudgetMiddleware(AgentMiddleware):
    """Make the investigation's last allowed turn produce findings.

    A plain model-call limit ends the loop with nothing to report, which turns a
    long investigation into a wasted one. This instead spends the final turn on
    the conclusion, so the phase always yields findings.
    """

    def __init__(
        self,
        max_model_calls: int,
        conclusion: ToolStrategy[Any] | ProviderStrategy[Any],
    ) -> None:
        super().__init__()
        self._budget = max_model_calls
        self._conclusion = conclusion

    def _bounded(self, request: ModelRequest) -> ModelRequest:
        turns = sum(1 for message in request.messages if isinstance(message, AIMessage))
        if turns < self._budget - 1:
            return request
        logger.info("investigation budget reached; forcing the conclusion")
        # The search tools go. Where the model can be forced, the answer comes
        # through the Findings tool: a request that still carries tool calls in
        # its transcript but declares no tools is one Bedrock rejects, and
        # langchain-aws then flattens the history to text with a warning. A
        # model that cannot be forced takes that flattening and answers in the
        # native schema.
        return request.override(
            tools=[],
            response_format=self._conclusion,
            messages=[*request.messages, HumanMessage(FINAL_TURN_NUDGE)],
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._bounded(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._bounded(request))


FINAL_TURN_NUDGE = (
    "This is the final turn of the investigation — no further searches will run. "
    "Submit your findings now with what the evidence so far supports."
)
