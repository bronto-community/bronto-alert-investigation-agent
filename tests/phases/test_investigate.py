"""Tests for investigation phase: window scoping, the model-call budget, the node's
handoff, and native structured output."""

import pytest
from langchain.agents.middleware import ModelRequest
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt.tool_node import ToolCallRequest

from investigator.phases.investigate import (
    FINAL_TURN_NUDGE,
    ConcludeWithinBudgetMiddleware,
    Findings,
    InvestigationWindowMiddleware,
    build_investigation_node,
    investigation_tools,
    schema_without_time_range,
    scope_time,
)
from investigator.settings import Deps

WINDOW = {"start_ms": 100, "end_ms": 200}
ALERT_CONTEXT = {"window": WINDOW}


def _request(name, args, state=None):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "c1"},
        tool=None,
        state=state if state is not None else {},
        runtime=None,
    )


def _capture():
    """An async handler recording the args it was handed (the live ToolNode path)."""
    seen: dict = {}

    async def handler(request):
        seen["args"] = request.tool_call["args"]
        return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])

    return handler, seen


def _capture_sync():
    """The sync equivalent, for the sync hook."""
    seen: dict = {}

    def handler(request):
        seen["args"] = request.tool_call["args"]
        return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])

    return handler, seen


# --- window scoping -----------------------------------------------------------


async def test_defaults_timerange_from_the_alert_context_window():
    """When the model omits the epoch range it is filled from the alert context."""
    handler, seen = _capture()

    await InvestigationWindowMiddleware().awrap_tool_call(
        _request(
            "search_logs", {"search_filter": "err"}, {"alert_context": ALERT_CONTEXT}
        ),
        handler,
    )

    assert seen["args"]["timerange_start"] == 100
    assert seen["args"]["timerange_end"] == 200
    assert seen["args"]["search_filter"] == "err"


def test_window_middleware_implements_the_sync_hook_too():
    handler, seen = _capture_sync()

    InvestigationWindowMiddleware().wrap_tool_call(
        _request("search_logs", {"time_range": "Last 20 minutes"}), handler
    )

    assert "time_range" not in seen["args"]


def test_scope_time_drops_the_relative_range():
    """time_range is now-anchored, so it is wrong for a historical alert."""
    args = scope_time(
        "search_logs", {"time_range": "Last 20 minutes", "q": "err"}, WINDOW
    )

    assert "time_range" not in args
    assert args["q"] == "err"


def test_scope_time_defaults_the_epoch_range():
    """An omitted range is filled from the alert context's investigation window."""
    args = scope_time("timeseries", {}, WINDOW)

    assert (args["timerange_start"], args["timerange_end"]) == (100, 200)


def test_scope_time_honors_an_explicit_range():
    """Default-when-omitted, not always-override."""
    args = scope_time("search_logs", {"timerange_start": 5, "timerange_end": 9}, WINDOW)

    assert (args["timerange_start"], args["timerange_end"]) == (5, 9)


def test_scope_time_leaves_non_time_aware_tools_alone():
    """get_keys takes no range, so nothing is injected."""
    assert scope_time("get_keys", {"log_id": "a"}, WINDOW) == {"log_id": "a"}


def test_scope_time_without_a_window_only_strips():
    """With no window there is nothing to default, but time_range still goes."""
    assert scope_time("search_logs", {"time_range": "Last 1 hour"}, None) == {}


# --- concluding within the budget ---------------------------------------------


@tool
def search_logs(q: str = "") -> str:
    """Search the logs."""
    return "results"


def _model_request(model, turns: int) -> ModelRequest:
    """A request whose transcript already contains `turns` completed model calls."""
    messages: list = [HumanMessage("alert_context")]
    for i in range(turns):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "search_logs", "args": {}, "id": f"c{i}"}],
            )
        )
        messages.append(ToolMessage(content="results", tool_call_id=f"c{i}"))
    return ModelRequest(model=model, messages=messages, tools=[search_logs])


def _capture_request():
    seen: dict = {}

    def handler(request):
        seen["request"] = request
        return "response"

    return handler, seen


def test_early_turns_are_left_untouched(make_model):
    """With budget to spare the request passes through as-is, so the search
    tools stay available and the prompt prefix stays byte-identical."""
    handler, seen = _capture_request()
    request = _model_request(make_model([]), turns=1)

    ConcludeWithinBudgetMiddleware(5, ToolStrategy(Findings)).wrap_model_call(
        request, handler
    )

    assert seen["request"] is request


def test_the_last_allowed_turn_asks_for_the_findings(make_model):
    """The nudge tells the model no further searches will run, so it submits
    what the evidence supports rather than stalling."""
    handler, seen = _capture_request()

    ConcludeWithinBudgetMiddleware(5, ToolStrategy(Findings)).wrap_model_call(
        _model_request(make_model([]), turns=4), handler
    )

    assert seen["request"].messages[-1].content == FINAL_TURN_NUDGE


@pytest.mark.parametrize(
    "budget,turns,forced",
    [(3, 0, False), (3, 1, False), (3, 2, True), (5, 4, True), (1, 0, True)],
)
def test_turns_are_counted_from_the_transcript(make_model, budget, turns, forced):
    """One AIMessage per model call, so the count needs no state channel and
    cannot drift or reset across an interrupt/resume. The last allowed turn has
    its search tools removed and answers through the Findings tool; a budget of
    one forces the very first turn."""
    handler, seen = _capture_request()

    ConcludeWithinBudgetMiddleware(budget, ToolStrategy(Findings)).wrap_model_call(
        _model_request(make_model([]), turns), handler
    )

    assert (seen["request"].tools == []) is forced
    assert isinstance(seen["request"].response_format, ToolStrategy) is forced


async def test_conclude_middleware_implements_the_async_hook_too(make_model):
    """The async path is the live one: the agent runs under ainvoke, so only
    awrap_model_call is ever called there."""
    seen: dict = {}

    async def handler(request):
        seen["request"] = request
        return "response"

    await ConcludeWithinBudgetMiddleware(2, ToolStrategy(Findings)).awrap_model_call(
        _model_request(make_model([]), turns=1), handler
    )

    assert seen["request"].tools == []
    assert isinstance(seen["request"].response_format, ToolStrategy)


# --- the node -----------------------------------------------------------------

# The scripted model has no profile unless given one, and then the bare
# response_format falls back to the tool strategy. NATIVE gives it the profile
# the real models carry, which resolves to the provider's native structured
# output instead.
NATIVE = {"structured_output": True}
FINDINGS = Findings(
    root_cause="GC pause",
    evidence=["p99 latency 4s at 10:02"],
    remediation=["raise heap"],
    confidence="medium",
)


def _node(model, tools=()):
    deps = Deps(client=None, model=model, investigation_tools=list(tools))
    return build_investigation_node(deps)


async def test_the_alert_context_window_reaches_the_agents_searches(make_model):
    """The writing half of the handoff. The window middleware reads the alert
    context from the agent's own state, so the node has to hand it over under
    the channel `DiagnosisState` declares. LangGraph drops an unknown key
    without a warning, and every search would then run unscoped."""
    scoped: list[tuple[int, int]] = []

    @tool
    async def search_logs(timerange_start: int = 0, timerange_end: int = 0) -> str:
        """Search the logs."""
        scoped.append((timerange_start, timerange_end))
        return "results"

    model = make_model(
        [
            AIMessage(
                content="", tool_calls=[{"name": "search_logs", "args": {}, "id": "c1"}]
            ),
            AIMessage(content=FINDINGS.model_dump_json()),
        ],
        profile=NATIVE,
    )

    await _node(model, [search_logs])({"alert_context": ALERT_CONTEXT})

    assert scoped == [(WINDOW["start_ms"], WINDOW["end_ms"])]


async def test_a_native_structured_answer_becomes_the_findings(make_model):
    """Natively, a reply with no tool calls is the answer: its JSON content is
    parsed into Findings rather than read as prose."""
    model = make_model([AIMessage(content=FINDINGS.model_dump_json())], profile=NATIVE)
    node = _node(model)

    assert await node({"alert_context": ALERT_CONTEXT}) == {
        "findings": FINDINGS.model_dump()
    }
    # The schema travels as a response format, so there is no Findings tool
    # and no forced tool choice for the model to work around.
    kwargs = model.call_kwargs[-1]
    assert "response_format" in kwargs
    assert "Findings" not in [getattr(t, "name", None) for t in model.bound_tools or []]
    assert kwargs.get("tool_choice") is None


async def test_the_forced_final_turn_still_declares_a_tool(make_model):
    """The last turn's search tools are dropped, but the transcript still holds
    tool calls and results. A request with that history and no tools declared
    is one Bedrock rejects, and langchain_aws then flattens the history to text
    with a warning. So the final turn answers through the Findings tool, forced."""

    @tool
    async def search_logs(q: str = "") -> str:
        """Search the logs."""
        return "results"

    model = make_model(
        [
            AIMessage(
                content="", tool_calls=[{"name": "search_logs", "args": {}, "id": "c1"}]
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "Findings", "args": FINDINGS.model_dump(), "id": "c2"}
                ],
            ),
        ],
        profile=NATIVE,
    )
    deps = Deps(client=None, model=model, investigation_tools=[search_logs])
    node = build_investigation_node(deps, max_model_calls=2)

    assert await node({"alert_context": ALERT_CONTEXT}) == {
        "findings": FINDINGS.model_dump()
    }
    assert [getattr(t, "name", None) for t in model.bound_tools or []] == ["Findings"]
    assert model.call_kwargs[-1]["tool_choice"] == "any"


async def test_a_model_that_cannot_be_forced_answers_natively_to_the_end(
    make_model,
):
    """Bedrock rejects tool_choice "any" outright for some models, and their
    profile does not claim native structured output, so the bare schema would
    resolve to the forced Findings tool. Every turn, the last included, asks
    for the native schema instead and never forces a tool."""

    @tool
    async def search_logs(q: str = "") -> str:
        """Search the logs."""
        return "results"

    model = make_model(
        [
            AIMessage(
                content="", tool_calls=[{"name": "search_logs", "args": {}, "id": "c1"}]
            ),
            AIMessage(content=FINDINGS.model_dump_json()),
        ],
        tool_choices=("auto",),
    )
    deps = Deps(client=None, model=model, investigation_tools=[search_logs])
    node = build_investigation_node(deps, max_model_calls=2)

    assert await node({"alert_context": ALERT_CONTEXT}) == {
        "findings": FINDINGS.model_dump()
    }
    for kwargs in model.call_kwargs:
        assert "response_format" in kwargs
        assert kwargs.get("tool_choice") is None
    assert "Findings" not in [getattr(t, "name", None) for t in model.bound_tools or []]


async def test_an_unparseable_native_answer_is_reported_as_no_findings(make_model):
    model = make_model([AIMessage(content="I think it was the GC.")], profile=NATIVE)
    node = _node(model)

    assert await node({"alert_context": ALERT_CONTEXT}) == {"findings": None}
    # Natively, not the tool fallback, where prose just ends the loop quietly.
    assert "response_format" in model.call_kwargs[-1]


# --- the tools it may call ----------------------------------------------------


def _tool(name: str, properties: tuple[str, ...] = ("q",)) -> StructuredTool:
    """A loaded MCP tool: a JSON-schema dict and a content_and_artifact coroutine."""

    async def call_tool(**kwargs):
        return "result", {"rows": []}

    return StructuredTool(
        name=name,
        description=f"The {name} tool.",
        args_schema={
            "type": "object",
            "properties": {prop: {"type": "string"} for prop in properties},
            "required": list(properties),
        },
        coroutine=call_tool,
        response_format="content_and_artifact",
    )


def test_investigation_tools_returns_the_four_in_order():
    """Only the 4 investigation tools are selected from the loaded MCP tools,
    in a stable order; everything else is dropped."""
    loaded = [
        _tool("search_monitors"),
        _tool("timeseries"),
        _tool("get_keys"),
        _tool("get_datasets"),
        _tool("search_logs"),
        _tool("get_key_values"),
    ]

    result = investigation_tools(loaded)

    assert [tool.name for tool in result] == [
        "get_keys",
        "get_key_values",
        "search_logs",
        "timeseries",
    ]


def test_investigation_tools_errors_on_missing_tool():
    """A required tool absent from the loaded set raises a clear error naming
    the missing tool."""
    loaded = [_tool("get_keys"), _tool("search_logs"), _tool("timeseries")]

    with pytest.raises(ValueError, match="get_key_values"):
        investigation_tools(loaded)


def test_investigation_tools_hides_time_range_but_keeps_the_tool_intact():
    """Rebuilding the tool as a fresh StructuredTool would drop the MCP
    content_and_artifact response format, losing every artifact. Copying it and
    replacing only the schema keeps the coroutine and format."""
    search_logs = _tool("search_logs", ("time_range", "q"))
    loaded = [
        _tool("get_keys"),
        _tool("get_key_values"),
        search_logs,
        _tool("timeseries"),
    ]

    narrowed = investigation_tools(loaded)[2]

    assert "time_range" not in narrowed.args_schema["properties"]
    assert narrowed.args_schema["required"] == ["q"]
    assert narrowed.name == "search_logs"
    assert narrowed.description == "The search_logs tool."
    assert narrowed.response_format == "content_and_artifact"
    assert narrowed.coroutine is search_logs.coroutine
    # The original is untouched.
    assert "time_range" in search_logs.args_schema["properties"]


def test_schema_without_time_range_drops_field_from_properties_and_required():
    """time_range is removed from the JSON schema's properties and required list;
    everything else is kept."""
    schema = {
        "type": "object",
        "properties": {
            "search_filter": {"type": "string"},
            "time_range": {"type": "string"},
            "log_ids": {"type": "array"},
        },
        "required": ["search_filter", "time_range"],
    }

    assert schema_without_time_range(schema) == {
        "type": "object",
        "properties": {
            "search_filter": {"type": "string"},
            "log_ids": {"type": "array"},
        },
        "required": ["search_filter"],
    }


def test_schema_without_time_range_is_a_no_op_without_the_field():
    """A schema with no time_range comes back as the same object."""
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    assert schema_without_time_range(schema) is schema
