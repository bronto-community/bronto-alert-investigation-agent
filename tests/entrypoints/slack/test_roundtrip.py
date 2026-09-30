"""The Slack human drives the real interrupt loop, and agrees with the terminal.

The same human intent expressed through Slack produces byte-identical
LangGraph decisions to the same intent typed into a terminal. If these two
ever disagree, one surface is lying to the graph.
"""

import asyncio

import pytest
from langgraph.types import Command

from investigator.entrypoints.cli.terminal import TerminalHuman
from investigator.entrypoints.slack.thread import SlackHuman
from investigator.human import Answer
from investigator.phases.identify.confirm import CONFIRM_ALERT
from investigator.run import run_config, run_to_completion
from tests.conftest import a_spine, confirm_alert_interrupt, mock_graph, until
from tests.entrypoints.slack.conftest import FakeSlackApi

_C0 = {"id": "m-1", "label": "checkout errors", "occurred_at": "2026-07-24T14:02:00Z"}
_C1 = {"id": "m-2", "label": "payment latency", "occurred_at": "2026-07-24T09:30:00Z"}


async def test_run_to_completion_runs_end_to_end_over_slack(mocker):
    """`ask` blocking on a Future that a Slack listener resolves is exactly
    what the loop already assumes of any human."""
    app = mock_graph(
        mocker,
        {"__interrupt__": [confirm_alert_interrupt(_C0, _C1)]},
        {"report": "GC pause."},
    )
    api = FakeSlackApi()
    human = SlackHuman(api, thread_ts="1700000000.000100", nonce="n1")

    run = asyncio.create_task(
        run_to_completion(app, a_spine(), human, config=run_config("slack:t:c:1.0"))
    )
    await until(lambda: human.pending_seq == 1)
    human.submit(seq=1, option_id="1")
    result = await run

    assert result["report"] == "GC pause."
    resumed = app.ainvoke.call_args_list[1].args[0]
    assert isinstance(resumed, Command)
    assert resumed.resume["decisions"] == [
        {
            "type": "edit",
            "edited_action": {"name": CONFIRM_ALERT, "args": {"candidates": [_C1]}},
        }
    ]
    # The report reached the thread rather than a terminal.
    assert any(p["blocks"][0]["type"] == "markdown" for p in api.posts)


async def _decision_via_slack(answer: Answer, mocker) -> dict:
    app = mock_graph(
        mocker,
        {"__interrupt__": [confirm_alert_interrupt(_C0, _C1)]},
        {"report": "done"},
    )
    human = SlackHuman(FakeSlackApi(), thread_ts="1.0", nonce="n")
    run = asyncio.create_task(
        run_to_completion(app, a_spine(), human, config=run_config("t"))
    )
    await until(lambda: human.pending_seq == 1)
    human.submit(seq=1, option_id=answer.option_id, text=answer.text, skip=answer.skip)
    await run
    return app.ainvoke.call_args_list[1].args[0].resume["decisions"][0]


async def _decision_via_terminal(typed: str, mocker) -> dict:
    app = mock_graph(
        mocker,
        {"__interrupt__": [confirm_alert_interrupt(_C0, _C1)]},
        {"report": "done"},
    )
    answers = iter([typed])
    human = TerminalHuman(input_fn=lambda: next(answers), output_fn=lambda _m: None)
    await run_to_completion(app, a_spine(), human, config=run_config("t"))
    return app.ainvoke.call_args_list[1].args[0].resume["decisions"][0]


@pytest.mark.parametrize(
    "answer,typed",
    [
        (Answer(option_id="1"), "1"),
        (Answer(text="try payments"), "try payments"),
        (Answer(option_id="0"), ""),
    ],
    ids=[
        "a click and a typed index",
        "a reply and typed text",
        "the default and a bare enter",
    ],
)
async def test_slack_and_the_terminal_mean_the_same_thing(mocker, answer, typed):
    """Both surfaces route through the same `to_decision`, so one intent is one
    decision, not two dialects of one."""
    assert await _decision_via_slack(answer, mocker) == await _decision_via_terminal(
        typed, mocker
    )
