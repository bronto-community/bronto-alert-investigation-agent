"""Shared test fixtures."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from bronto_sdk.models import (
    DatasetsResponse,
    MetricDefinition,
    Monitor,
    SearchResponse,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from investigator.human import Answer, Event, Question
from investigator.phases.identify.confirm import CONFIRM_ALERT
from investigator.phases.identify.questions import confirm_alert_question
from investigator.state import InvestigationState, state_for_request


class ScriptedChatModel(BaseChatModel):
    """Scriptable, network-free BaseChatModel for tests.

    A real ``BaseChatModel`` subclass, so it can be handed to ``create_agent``
    (which drives the model through Runnable machinery and calls ``bind_tools``
    on every model call). Pops pre-scripted AIMessages in order and records the
    prompts and per-call kwargs, so tests can assert on prompt assembly, tool
    binding, and per-call provider kwargs.
    """

    responses: list[BaseMessage] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    call_kwargs: list[dict] = Field(default_factory=list)
    # Mirrors the attribute langchain-aws sets; None is a model that can force.
    supports_tool_choice_values: tuple[str, ...] | None = None

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=list(tools), **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(list(messages))
        self.call_kwargs.append(dict(kwargs))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._generate(messages, stop, None, **kwargs)

    @property
    def bound_tools(self) -> list | None:
        """The tool list from the most recent model call, or None if unbound."""
        for call in reversed(self.call_kwargs):
            if "tools" in call:
                return call["tools"]
        return None


@pytest.fixture
def make_model():
    """Return a factory building a ScriptedChatModel from scripted responses."""

    def _make(
        responses: list[BaseMessage] | None = None,
        profile: dict | None = None,
        tool_choices: tuple[str, ...] | None = None,
    ) -> ScriptedChatModel:
        """``profile={"structured_output": True}`` makes a bare response_format
        resolve to the provider's native structured output, as it does for the
        real models; without it the agent falls back to the tool strategy.
        ``tool_choices=("auto",)`` is a model that rejects forced tool use."""
        return ScriptedChatModel(
            responses=list(responses or []),
            profile=profile,
            supports_tool_choice_values=tool_choices,
        )

    return _make


def a_monitor(**fields) -> Monitor:
    """The monitor `mon-1`, as `GET /monitors/{id}` returns it: no queries."""
    return Monitor.model_validate(
        {
            "id": "mon-1",
            "name": "foo bar baz",
            "monitor_type": "PATTERN",
            "window": "last 30 minutes",
            "metric_id": "md-1",
            **fields,
        }
    )


@pytest.fixture
def fake_bronto(mocker):
    """A BrontoClient double over one monitor, `mon-1`, and one dataset, `aaa`.

    Its audit search grounds one triggered alert, and its typed lookups serve
    context generation: the monitor, the metric definition holding its query,
    the dataset that query reads, and a flat state timeseries. Built from the
    SDK's own models, so a field renamed there breaks here rather than passing.

    The query carries an `aggregation` key no consumer reads: it is there so
    trimming to the select/where/groups whitelist is actually exercised rather
    than being a no-op. The monitor has no configured playbook.
    """
    client = mocker.Mock()
    client.post.return_value = {
        "result": [
            {
                "timestamp": "2026-07-24T14:19:09.453550458Z",
                "service_payload.monitor_id": "mon-1",
                "service_payload.monitor_name": "foo bar baz",
                "service_payload.evaluation_timestamp": "1784901720000",
                "service_payload.new_status": "ALERT",
            }
        ]
    }
    client.monitors.retrieve.return_value = a_monitor()
    client.metrics.retrieve_definition.return_value = MetricDefinition.model_validate(
        {
            "id": "md-1",
            "queries": [
                {
                    "select": ["COUNT(*)"],
                    "from": ["aaa"],
                    "from_expr": "",
                    "where": "'foo bar'",
                    "groups": [],
                    "aggregation": [{"time": "count", "reduce_to": "avg"}],
                }
            ],
        }
    )
    client.datasets.list.return_value = DatasetsResponse.model_validate(
        {"datasets": [{"id": "aaa", "dataset": "syslog", "collection": "my-service"}]}
    )
    client.search.run.return_value = SearchResponse.model_validate(
        {
            "groups_series": [
                {
                    "name": "",
                    "timeseries": [{"@timestamp": 1, "count": 5, "value": 5.0}],
                }
            ]
        }
    )
    return client


def a_spine(**overrides) -> InvestigationState:
    """A complete investigation state, for tests that mock the graph and never read it.

    Passing `{}` would be simpler but the parameter is a TypedDict, so a
    partial dict is a type error for a reason: a real graph would KeyError on
    the first missing field.
    """
    return {**state_for_request("oom on checkout"), **overrides}


def mock_graph(mocker, *returns):
    """A compiled-graph double whose successive ``ainvoke`` calls return ``returns``."""
    graph = mocker.Mock()
    graph.ainvoke = mocker.AsyncMock()
    graph.ainvoke.side_effect = list(returns)
    return graph


def confirm_alert_interrupt(*candidates):
    """A confirm-alert interrupt, shaped as HumanInTheLoopMiddleware produces one."""
    return SimpleNamespace(
        value={
            "action_requests": [
                {
                    "name": CONFIRM_ALERT,
                    "args": {"candidates": list(candidates)},
                    "description": confirm_alert_question(
                        {"args": {"candidates": list(candidates)}}, {}, None
                    ),
                }
            ]
        }
    )


async def until(predicate, *, ticks: int = 200):
    """Yield to the loop until `predicate` holds.

    A bounded spin rather than a sleep: the loop is single-threaded and there
    is no wall clock involved, so a test that waits on a Future being created
    or resolved is deterministic rather than merely usually-fast.
    """
    for _ in range(ticks):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never held")


class AutoHuman:
    """A ``Human`` for tests that drive a run end to end.

    Takes the default answer to everything, and keeps what it was asked and
    told so a test can assert on both.
    """

    def __init__(self) -> None:
        self.questions: list[Question] = []
        self.events: list[Event] = []

    async def ask(self, question: Question) -> Answer:
        self.questions.append(question)
        if not question.options and question.allow_skip:
            # Nothing proposed and nothing required: decline rather than fail.
            return Answer(skip=True)
        return Answer()

    async def tell(self, event: Event) -> None:
        self.events.append(event)

    @property
    def report(self) -> str | None:
        """The text of the first report told, or None if there was none."""
        return next((e.text for e in self.events if e.type == "report"), None)
