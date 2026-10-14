"""Tests for the unattended run registry."""

import asyncio

import pytest

from investigator.entrypoints.webhook.runs import AtCapacity, EventRuns
from investigator.human import Event
from tests.conftest import until

EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "evaluation_timestamp": 1784901720000,
}


class Recorder:
    """A `Human` with nothing to say and nowhere to say it."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def ask(self, question):
        raise AssertionError("an event run must never ask")

    async def tell(self, event: Event) -> None:
        self.events.append(event)


def _graph(mocker, result=None, *, hang: bool = False):
    graph = mocker.Mock()
    if hang:

        async def never(*args, **kwargs):
            await asyncio.Event().wait()

        graph.ainvoke = never
    else:
        graph.ainvoke = mocker.AsyncMock(return_value=result or {"report": "done"})
    return graph


def _runs(mocker, human, **kwargs):
    return EventRuns(_graph(mocker, **kwargs), human_for=lambda: human)


# --- one run ------------------------------------------------------------------


async def test_a_run_reports_through_the_human_it_was_given(mocker):
    human = Recorder()
    runs = _runs(mocker, human)

    runs.start(EVENT)
    await until(lambda: [e for e in human.events if e.type == "report"])

    assert human.events[0].text == "done"
    await runs.shutdown()


async def test_the_run_is_keyed_by_the_alert_not_the_monitor(mocker):
    """Two firings of one monitor are two investigations, so two checkpoints."""
    human = Recorder()
    graph = _graph(mocker)
    runs = EventRuns(graph, human_for=lambda: human)

    runs.start(EVENT)
    runs.start({**EVENT, "evaluation_timestamp": 1784901780000})
    await until(lambda: graph.ainvoke.call_count == 2)

    threads = {
        c.args[1]["configurable"]["thread_id"] for c in graph.ainvoke.call_args_list
    }
    assert threads == {
        "webhook:mon-1@1784901720000",
        "webhook:mon-1@1784901780000",
    }
    await runs.shutdown()


async def test_a_finished_run_gives_its_slot_back(mocker):
    human = Recorder()
    runs = EventRuns(_graph(mocker), human_for=lambda: human, max_live=1)

    runs.start(EVENT)
    await until(lambda: runs.live == 0)

    runs.start(EVENT)  # no raise: the first one is done
    await runs.shutdown()


# --- the cap ------------------------------------------------------------------


async def test_beyond_the_cap_it_refuses_rather_than_queues(mocker):
    """A queue a restart would silently drop is worse than an honest refusal."""
    graph = _graph(mocker, hang=True)
    runs = EventRuns(graph, human_for=Recorder, max_live=1)

    runs.start(EVENT)
    with pytest.raises(AtCapacity):
        runs.start({**EVENT, "evaluation_timestamp": 2})

    assert runs.live == 1
    await runs.shutdown()


# --- when a run goes wrong ----------------------------------------------------


async def test_a_wedged_run_is_abandoned_rather_than_holding_its_slot(mocker):
    human = Recorder()
    runs = EventRuns(
        _graph(mocker, hang=True), human_for=lambda: human, run_timeout=0.01
    )

    runs.start(EVENT)
    # The timeout is wall clock, so wait on the task rather than counting
    # loop ticks: how many ticks fit in 10ms depends on the machine.
    (task,) = runs._live
    await asyncio.wait_for(task, timeout=5)
    await until(lambda: runs.live == 0)

    assert any("too long" in e.text for e in human.events)


async def test_a_failing_run_says_so_and_frees_its_slot(mocker):
    human = Recorder()
    graph = _graph(mocker)
    graph.ainvoke = mocker.AsyncMock(side_effect=RuntimeError("bedrock said no"))
    runs = EventRuns(graph, human_for=lambda: human)

    runs.start(EVENT)
    await until(lambda: runs.live == 0)

    assert any("failed" in e.text for e in human.events)


async def test_shutdown_cancels_what_is_still_running(mocker):
    runs = EventRuns(_graph(mocker, hang=True), human_for=Recorder)

    runs.start(EVENT)
    tasks = list(runs._live)
    await runs.shutdown()

    assert tasks and all(task.cancelled() for task in tasks)
