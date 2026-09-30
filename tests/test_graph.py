"""Tests for the top-level orchestrator graph."""

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import Command

from investigator.graph import (
    GIVE_UP_MESSAGE,
    build_app,
    build_checkpointer,
    give_up,
    route_after_identification,
    route_entry,
)
from investigator.phases.identify.confirm import CONFIRM_ALERT, CONFIRM_PLAYBOOK
from investigator.phases.investigate import FINAL_TURN_NUDGE, Findings
from investigator.run import run_config, run_to_completion
from investigator.settings import Deps
from investigator.state import state_for_event, state_for_request
from tests.conftest import AutoHuman, a_monitor


def _app(tools, client, model, max_model_calls=5):
    """Build the orchestrator the way the CLI does, with an in-memory checkpointer."""
    return build_app(
        Deps(client=client, model=model, investigation_tools=tools),
        max_model_calls=max_model_calls,
        checkpointer=build_checkpointer(),
    )


def _submit_findings(root_cause, call_id="s1"):
    """An investigator turn that concludes by calling the structured-output tool."""
    findings = Findings(
        root_cause=root_cause,
        evidence=["foo bar baz"],
        remediation=["increase heap"],
        confidence="medium",
    )
    return AIMessage(
        content="",
        tool_calls=[{"name": "Findings", "args": findings.model_dump(), "id": call_id}],
    )


def test_build_app_assembles_one_node_per_phase(mocker, make_model):
    """Each phase is one node on the investigation state: the two agentic phases
    behind wrapper nodes, so their Reason/Act loops stay namespaced inside their
    agents, plus the deterministic context and report steps."""
    graph = _app([], mocker.Mock(), make_model([])).get_graph()

    assert set(graph.nodes) == {
        "__start__",
        "identification",
        "give_up",
        "context",
        "investigation",
        "report_generation",
        "__end__",
    }
    assert ("report_generation", "__end__") in {
        (e.source, e.target) for e in graph.edges
    }


async def test_orchestrator_runs_full_pipeline(mocker, make_model, fake_bronto):
    """A confirmed alert flows identification -> context -> investigation:
    alert_context is built and findings are produced on the investigation state."""
    client = fake_bronto
    model = make_model(
        [
            _search_audit(),
            _propose_mon1(),  # -> confirm_alert interrupt
            _ask_playbook(),  # -> confirm_playbook interrupt, then the phase ends
            _submit_findings("GC pause"),  # investigation concludes
            AIMessage(content="Human-readable incident report."),  # report_generation
        ],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "orch-ok"}}
    app = _app([], client, model)

    await app.ainvoke(state_for_request("OOM on host"), config)
    await app.ainvoke(APPROVE, config)  # confirm the one candidate
    result = await app.ainvoke(SKIP_PLAYBOOK, config)

    fake_bronto.metrics.retrieve_definition.assert_called_once_with("md-1")
    # The grounded audit row reached the alert context, not the lean candidate summary.
    detail = result["alert_context"]["detail"]
    assert detail["triggered_event"]["evaluation_timestamp"] == 1784901720000
    assert result["findings"]["root_cause"] == "GC pause"
    assert result["report"] == "Human-readable incident report."
    assert model.responses == []


async def test_orchestrator_retry_sends_identification_back_to_search(
    mocker, make_model, fake_bronto
):
    """Free text at the confirm prompt is a `respond` decision: the tool body is
    skipped, the model searches again, and the phase interrupts afresh — without
    re-running the searches already done."""
    client = fake_bronto
    model = make_model(
        [
            _search_audit(),
            _propose_mon1(),  # -> interrupt, rejected
            _search_audit(),  # searches again on the correction
            _propose_mon1(),  # -> interrupt, accepted
            _ask_playbook(),
            _submit_findings("GC pause"),
            AIMessage(content="Report."),
        ],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "orch-retry"}}
    app = _app([], client, model)

    await app.ainvoke(state_for_request("OOM on host"), config)
    await app.ainvoke(
        Command(resume={"decisions": [{"type": "respond", "message": "no, disk"}]}),
        config,
    )
    await app.ainvoke(APPROVE, config)
    result = await app.ainvoke(SKIP_PLAYBOOK, config)

    assert result["findings"]["root_cause"] == "GC pause"
    assert model.responses == []
    # The correction reached the model, framed as the rejection it is.
    assert any("no, disk" in str(m.content) for m in model.calls[2])


async def test_orchestrator_gives_up_on_an_agent_that_answers_in_prose(
    mocker, make_model, fake_bronto
):
    """An agent that answers instead of calling a tool ends its phase with no
    triggered_alert, so the pipeline routes to give_up and skips context generation
    rather than building an alert context from no alert."""
    client = fake_bronto
    model = make_model([AIMessage(content="I am not sure which alert.")])
    config: RunnableConfig = {"configurable": {"thread_id": "orch-giveup"}}
    app = _app([], client, model)

    result = await app.ainvoke(state_for_request("something vague"), config)

    assert len(model.calls) == 1
    assert result["triggered_alert"] is None
    assert result.get("alert_context") is None
    assert result["report"] == GIVE_UP_MESSAGE
    fake_bronto.metrics.retrieve_definition.assert_not_called()


async def test_orchestrator_gives_up_when_identification_never_stops_searching(
    mocker, make_model, fake_bronto
):
    client = fake_bronto
    client.post.return_value = {"result": []}  # every search comes back empty
    scripted = 40  # more turns than the loop's recursion bound can spend
    model = make_model([_search_audit() for _ in range(scripted)])
    config: RunnableConfig = {"configurable": {"thread_id": "orch-runaway"}}
    app = _app([], client, model)

    result = await app.ainvoke(state_for_request("something vague"), config)

    # Bounded by RECURSION_LIMIT rather than by exhausting the script.
    assert 0 < len(model.calls) < scripted
    assert result["triggered_alert"] is None
    assert result["report"] == GIVE_UP_MESSAGE
    fake_bronto.metrics.retrieve_definition.assert_not_called()


def _tool_names(tools) -> list[str]:
    """Names of bound tools, whichever shape the binding left them in."""
    return [t["function"]["name"] if isinstance(t, dict) else t.name for t in tools]


def _propose_mon1():
    """An identification turn proposing one candidate, which the HITL middleware
    interrupts on before the tool body runs."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": CONFIRM_ALERT,
                "args": {
                    "candidates": [
                        {
                            "id": "mon-1",
                            "label": "foo bar baz",
                            "occurred_at": "2026-07-24T14:19:09Z",
                        }
                    ]
                },
                "id": "p1",
            }
        ],
    )


def _ask_playbook():
    """The agent's terminating turn: ask for a playbook, then the phase ends.

    The model proposes null: the configured playbook is fetched after its turn,
    so it has never seen one to pass.
    """
    return AIMessage(
        content="",
        tool_calls=[
            {"name": CONFIRM_PLAYBOOK, "args": {"playbook": None}, "id": "pb1"}
        ],
    )


APPROVE = Command(resume={"decisions": [{"type": "approve"}]})
SKIP_PLAYBOOK = Command(
    resume={
        "decisions": [
            {
                "type": "edit",
                "edited_action": {"name": CONFIRM_PLAYBOOK, "args": {"playbook": None}},
            }
        ]
    }
)


def _search_audit():
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_audit_events",
                "args": {
                    "start": "2026-07-24T00:00:00Z",
                    "end": "2026-07-25T00:00:00Z",
                },
                "id": "t1",
            }
        ],
    )


def _investigate_call(call_id):
    return AIMessage(
        content="", tool_calls=[{"name": "search_logs", "args": {}, "id": call_id}]
    )


async def test_orchestrator_still_concludes_when_the_budget_runs_out(
    mocker, make_model, fake_bronto
):
    """A budget-exhausted investigation is not a wasted one: the last allowed
    turn is spent on the conclusion, so the run ends with real findings and a
    real report rather than the inconclusive notice."""
    client = fake_bronto
    model = make_model(
        [
            _search_audit(),
            _propose_mon1(),
            _ask_playbook(),
            _investigate_call("i1"),  # turn 1: still searching
            _submit_findings("unclear", "s1"),  # turn 2: forced to conclude
            AIMessage(content="Report after the forced conclusion."),
        ],
    )

    @tool
    async def search_logs(search_filter: str = "") -> str:
        """Search the logs."""
        return "log results"

    config: RunnableConfig = {"configurable": {"thread_id": "orch-limit"}}
    app = _app([search_logs], client, model, max_model_calls=2)

    await app.ainvoke(state_for_request("OOM on host"), config)
    await app.ainvoke(APPROVE, config)
    result = await app.ainvoke(SKIP_PLAYBOOK, config)

    assert result["findings"]["root_cause"] == "unclear"
    assert result["report"] == "Report after the forced conclusion."
    assert model.responses == []
    # The script would conclude on turn 2 regardless, so check it was forced:
    # the nudge was sent and the search tool was withheld on that call.
    forced = 4  # search, propose, ask playbook, investigate, then this one
    assert model.calls[forced][-1].content == FINAL_TURN_NUDGE
    assert "search_logs" not in _tool_names(model.call_kwargs[forced]["tools"])


async def test_the_alert_context_window_reaches_every_search(
    mocker, make_model, fake_bronto
):
    scoped: list[dict] = []

    @tool
    async def search_logs(
        search_filter: str = "", timerange_start: int = 0, timerange_end: int = 0
    ) -> str:
        """Search the logs."""
        scoped.append({"start_ms": timerange_start, "end_ms": timerange_end})
        return "log results"

    model = make_model(
        [
            _search_audit(),
            _propose_mon1(),
            _ask_playbook(),
            _investigate_call("i1"),
            _submit_findings("GC pause", "s1"),
            AIMessage(content="The report."),
        ],
    )
    config: RunnableConfig = {"configurable": {"thread_id": "orch-window"}}
    app = _app([search_logs], fake_bronto, model)

    await app.ainvoke(state_for_request("OOM on host"), config)
    await app.ainvoke(APPROVE, config)
    result = await app.ainvoke(SKIP_PLAYBOOK, config)

    assert scoped == [result["alert_context"]["window"]]


EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
    "new_status": "ALERT",
}


async def test_an_event_run_spends_no_identification_call(
    mocker, make_model, fake_bronto
):
    """No identification model call is spent identifying what is already known: the
    script here holds exactly the two calls the later phases make."""
    model = make_model(
        [
            _submit_findings("GC pause"),
            AIMessage(content="Human-readable incident report."),
        ]
    )
    app = _app([], fake_bronto, model)

    result = await app.ainvoke(state_for_event(EVENT), run_config("event-1"))

    assert result["findings"]["root_cause"] == "GC pause"
    assert result["report"] == "Human-readable incident report."
    # Two scripted calls, both consumed: the investigation's conclusion and the
    # report. An identification turn would have needed a third.
    assert model.responses == []
    # No audit search ran — the triggered alert was already grounded.
    fake_bronto.post.assert_not_called()


def test_routing_continues_only_with_a_confirmed_alert():
    """No triggered alert means identification never grounded one, which is a give-up
    rather than an alert context built from nothing."""
    assert route_after_identification({"triggered_alert": {"id": "m-1"}}) == "context"
    assert route_after_identification({"triggered_alert": None}) == "give_up"


def test_entry_routes_a_triggered_alert_straight_to_context():
    """A run fed an event has nothing to identify."""
    assert route_entry({"triggered_alert": {"id": "m-1"}}) == "context"


def test_entry_routes_a_freeform_request_to_identification():
    assert route_entry({"triggered_alert": None}) == "identification"


def test_give_up_writes_a_report():
    """Giving up is a terminal path like any other, so it still produces the
    one field every surface reads."""
    assert give_up({}) == {"report": GIVE_UP_MESSAGE}


async def test_accepting_the_default_playbook_records_the_configured_one(
    make_model, fake_bronto
):
    fake_bronto.monitors.retrieve.return_value = a_monitor(
        ai_report_instructions="check redis"
    )
    model = make_model(
        [
            _search_audit(),
            _propose_mon1(),
            _ask_playbook(),
            # With a playbook, context generation derives related datasets.
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "DeriveDatasets", "args": {"datasets": []}, "id": "d1"}
                ],
            ),
            _submit_findings("GC pause"),
            AIMessage(content="Report."),
        ],
    )
    app = _app([], fake_bronto, model)
    human = AutoHuman()

    result = await run_to_completion(
        app, state_for_request("OOM on host"), human, config=run_config("orch-pb")
    )

    assert human.questions[1].options[0].subtitle == "check redis"
    assert result["playbook"] == "check redis"
    assert result["alert_context"]["playbook"] == "check redis"
