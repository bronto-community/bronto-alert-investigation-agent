"""Tests for report generation."""

import logging

from langchain_core.messages import AIMessage

from investigator.phases.report import INCONCLUSIVE_REPORT, make_report_node


def _state(alert_context=None, findings=None, playbook=None):
    return {
        "alert_context": {
            "window": {"start_ms": 1, "end_ms": 2},
            "playbook": playbook,
            "detail": alert_context or {"monitor": {"name": "foo bar baz"}},
        },
        "findings": findings,
    }


async def test_report_node_makes_one_plain_model_call(make_model):
    """Report generation is a single untooled call, not another agent loop, and
    its text is what the run tells the human."""
    model = make_model([AIMessage(content="The monitor fired because of a GC pause.")])

    delta = await make_report_node(model)(_state(findings={"root_cause": "GC pause"}))

    assert len(model.calls) == 1
    assert model.bound_tools is None
    assert delta == {"report": "The monitor fired because of a GC pause."}


async def test_report_prompt_carries_alert_context_playbook_and_findings(make_model):
    """The call is given the alert context, the playbook the user chose, and the
    findings."""
    model = make_model([AIMessage(content="report")])

    await make_report_node(model)(
        _state(
            alert_context={"monitor": {"name": "foo bar baz"}},
            findings={"root_cause": "GC pause"},
            playbook="check redis",
        )
    )

    sent = str(model.calls[0])
    assert "foo bar baz" in sent
    assert "check redis" in sent
    assert "GC pause" in sent


async def test_no_findings_reports_inconclusively_without_a_model_call(
    make_model, caplog
):
    """A run cut off by the model-call limit has no findings to report from.
    Prompting the model over the raw transcript instead would be a best-effort
    extraction with nothing to ground it, so say so plainly."""
    model = make_model([])

    with caplog.at_level(logging.WARNING, logger="investigator.phases.report"):
        delta = await make_report_node(model)(_state(findings=None))

    assert delta["report"] == INCONCLUSIVE_REPORT
    assert model.calls == []
    assert "without findings" in caplog.text
