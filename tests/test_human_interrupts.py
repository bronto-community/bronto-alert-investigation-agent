"""The Question survives a real interrupt unchanged."""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from investigator.human import Answer, Question, to_decision
from investigator.phases.identify.confirm import CONFIRM_ALERT, IdentificationState
from investigator.phases.identify.identify import (
    CONFIRM_DESCRIPTION,
)
from investigator.phases.identify.questions import INTERRUPT_ON

CANDIDATES = [
    {
        "id": "mon-1",
        "label": "foo bar baz",
        "occurred_at": "2026-07-24T14:19:09Z",
    },
    {"id": "mon-2", "label": "latency", "occurred_at": "2026-07-24T09:30:00Z"},
]

# The rows a search retained as its artifact. A confirmed candidate has to link
# back to one of these or the tool rejects it as invented.
ROWS = [
    {"monitor_id": c["id"], "triggered_at": c["occurred_at"], "evaluation_timestamp": 1}
    for c in CANDIDATES
]


def _searched():
    """A transcript in which the candidates have actually been retrieved."""
    return [
        HumanMessage("foo"),
        ToolMessage(content="[]", tool_call_id="t0", artifact=ROWS),
    ]


def _propose():
    return AIMessage(
        content="",
        tool_calls=[
            {"name": CONFIRM_ALERT, "args": {"candidates": CANDIDATES}, "id": "c1"}
        ],
    )


def _agent(make_model, *extra):
    from investigator.phases.identify.confirm import (
        make_confirm_alert_tool,
    )

    confirm = make_confirm_alert_tool(
        description=CONFIRM_DESCRIPTION,
    )
    return create_agent(
        make_model([_propose(), *extra]),
        [confirm],
        state_schema=IdentificationState,
        middleware=[HumanInTheLoopMiddleware(INTERRUPT_ON)],
        checkpointer=MemorySaver(),
    )


async def test_a_question_reaches_the_interrupt_payload_as_a_dict(make_model):
    agent = _agent(make_model)
    config = {"configurable": {"thread_id": "rt-1"}}

    result = await agent.ainvoke({"messages": _searched()}, config)

    request = result["__interrupt__"][0].value["action_requests"][0]
    # Not a string, and not a model instance — a plain dict, which is what
    # survives a persistent checkpointer's serializer.
    assert isinstance(request["description"], dict)

    question = Question.model_validate(request["description"])
    assert question.action == CONFIRM_ALERT
    assert [o.label for o in question.options] == [
        "foo bar baz",
        "latency",
    ]


async def test_a_decision_built_from_that_question_actually_resumes(make_model):
    """The full loop: interrupt -> Question -> Answer -> decision -> resume, with
    the real middleware applying the decision rather than a test double."""
    # One more turn after the tool runs: confirm_alert is not return_direct.
    agent = _agent(make_model, AIMessage(content="confirmed"))
    config = {"configurable": {"thread_id": "rt-2"}}

    result = await agent.ainvoke({"messages": _searched()}, config)
    question = Question.model_validate(
        result["__interrupt__"][0].value["action_requests"][0]["description"]
    )

    decision = to_decision(question, Answer(option_id="1"))
    resumed = await agent.ainvoke(Command(resume={"decisions": [decision]}), config)

    # The middleware narrowed the proposal to the chosen candidate, and the
    # tool body then grounded it — failing here would mean the edit was dropped.
    assert resumed["triggered_alert"]["id"] == "mon-2"
