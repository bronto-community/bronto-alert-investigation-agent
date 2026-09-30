"""Tests for the interrupt loop, driven through a Human rather than stdin."""

import logging
from types import SimpleNamespace

from langgraph.types import Command

from investigator.human import Answer, Question
from investigator.phases.identify.confirm import CONFIRM_ALERT
from investigator.run import run_config, run_to_completion
from tests.conftest import a_spine, confirm_alert_interrupt, mock_graph

_C0 = {"id": "m-1", "label": "checkout errors", "occurred_at": "2026-07-24T14:02:00Z"}
_C1 = {"id": "m-2", "label": "payment latency", "occurred_at": "2026-07-24T09:30:00Z"}


class ScriptedHuman:
    """Answers from a fixed list; records the questions it was asked."""

    def __init__(self, *answers: Answer) -> None:
        self._answers = list(answers)
        self.questions: list[Question] = []
        self.events: list = []

    async def ask(self, question: Question) -> Answer:
        self.questions.append(question)
        return self._answers.pop(0)

    async def tell(self, event) -> None:
        self.events.append(event)


async def test_a_run_with_no_interrupt_just_reports(mocker, caplog):
    """The report is told rather than printed, so any surface can post it;
    bookkeeping goes to the log, never to the human."""
    app = mock_graph(mocker, {"findings": {"root_cause": "GC"}, "report": "the report"})
    human = ScriptedHuman()

    with caplog.at_level(logging.INFO, logger="investigator.run"):
        result = await run_to_completion(app, a_spine(), human, config=run_config("t"))

    assert result["report"] == "the report"
    assert human.questions == []
    assert [(e.type, e.text) for e in human.events] == [("report", "the report")]
    assert "investigation finished" in caplog.text


async def test_an_interrupt_is_asked_and_the_decision_resumes_the_graph(mocker):
    app = mock_graph(
        mocker, {"__interrupt__": [confirm_alert_interrupt(_C0)]}, {"report": "done"}
    )
    human = ScriptedHuman(Answer())

    await run_to_completion(app, a_spine(), human, config=run_config("t"))

    assert human.questions[0].prompt == "Which one are you investigating?"
    assert app.ainvoke.call_count == 2
    resumed = app.ainvoke.call_args_list[1].args[0]
    assert isinstance(resumed, Command)
    assert resumed.resume == {
        "decisions": [
            {
                "type": "edit",
                "edited_action": {
                    "name": CONFIRM_ALERT,
                    "args": {"candidates": [_C0]},
                },
            }
        ]
    }


async def test_every_action_request_gets_exactly_one_decision(mocker):
    """The middleware rejects a count mismatch."""
    payload = SimpleNamespace(
        value={
            "action_requests": [
                confirm_alert_interrupt(_C0).value["action_requests"][0],
                confirm_alert_interrupt(_C1).value["action_requests"][0],
            ]
        }
    )
    app = mock_graph(mocker, {"__interrupt__": [payload]}, {"report": "done"})

    await run_to_completion(
        app, a_spine(), ScriptedHuman(Answer(), Answer()), config=run_config("t")
    )

    assert len(app.ainvoke.call_args_list[1].args[0].resume["decisions"]) == 2


async def test_a_plain_string_description_degrades_to_a_free_text_question(mocker):
    """A tool that never opted into structured questions still
    works, rather than failing the run."""
    raw = SimpleNamespace(
        value={
            "action_requests": [
                {"name": "confirm_thing", "args": {}, "description": "Which thing?"}
            ]
        }
    )
    app = mock_graph(mocker, {"__interrupt__": [raw]}, {"report": "done"})
    human = ScriptedHuman(Answer(text="that one"))

    await run_to_completion(app, a_spine(), human, config=run_config("t"))

    assert human.questions[0].prompt == "Which thing?"
    assert app.ainvoke.call_args_list[1].args[0].resume["decisions"][0] == {
        "type": "respond",
        "message": "that one",
    }
