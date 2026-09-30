"""Tests for the run registry: task lifecycle, the cap, and the timeouts."""

import asyncio

from investigator.entrypoints.slack.runs import RunRegistry, thread_id_for
from tests.conftest import confirm_alert_interrupt, mock_graph, until
from tests.entrypoints.slack.conftest import FakeSlackApi

_PENDING = {
    "__interrupt__": [
        confirm_alert_interrupt({"id": "m-1", "label": "checkout errors"})
    ]
}


def _registry(graph, **kwargs) -> RunRegistry:
    return RunRegistry(graph, **kwargs)


def _start(registry, api, *, thread_ts="1.0", text="oom on checkout", **kwargs):
    return registry.start(
        api,
        team_id="T1",
        channel_id="C1",
        thread_ts=thread_ts,
        request_text=text,
        **kwargs,
    )


# --- thread ids ---------------------------------------------------------------


def test_the_thread_id_is_scoped_to_the_channel():
    """A ts is only unique within a channel, and the checkpointer is
    process-global, so two threads in different channels must not collide."""
    a = thread_id_for("T1", "C1", "1700000000.1")
    b = thread_id_for("T1", "C2", "1700000000.1")

    assert a != b
    assert a == "slack:T1:C1:1700000000.1"


# --- lifecycle ----------------------------------------------------------------


async def test_a_run_drives_the_graph_and_clears_itself_up(mocker):
    registry = _registry(mock_graph(mocker, {"report": "GC pause."}), max_live=1)
    api = FakeSlackApi()

    handle = _start(registry, api)
    await handle.task

    assert registry.get("C1", "1.0") is None, "a finished run must free its slot"
    assert registry.can_start() is None
    assert any("GC pause." in p["blocks"][0].get("text", "") for p in api.posts)


async def test_a_live_run_is_findable_by_its_thread(mocker):
    graph = mock_graph(mocker, _PENDING, {"report": "done"})
    registry = _registry(graph)
    handle = _start(registry, FakeSlackApi())
    await until(lambda: handle.human.pending_seq == 1)

    assert registry.get("C1", "1.0") is handle
    assert registry.get("C1", "9.9") is None
    await registry.shutdown()


async def test_an_empty_trigger_asks_for_a_description_first(mocker):
    """A bare `/investigate` is not an error: the run asks the same question the
    CLI asks, through the same human."""
    graph = mock_graph(mocker, {"report": "done"})
    registry = _registry(graph)
    api = FakeSlackApi()

    handle = _start(registry, api, text="")
    await until(lambda: handle.human.pending_seq == 1)
    assert "Describe what" in api.posts[0]["blocks"][0]["text"]["text"]

    handle.human.submit(seq=1, text="oom on checkout")
    await handle.task

    assert graph.ainvoke.call_args.args[0]["request_text"] == "oom on checkout"


# --- the cap ------------------------------------------------------------------


async def test_the_registry_refuses_to_start_past_its_cap(mocker):
    graph = mock_graph(mocker, *[_PENDING] * 2)
    registry = _registry(graph, max_live=2)
    api = FakeSlackApi()
    a = _start(registry, api, thread_ts="1.0")
    b = _start(registry, api, thread_ts="2.0")
    await until(lambda: a.human.pending_seq == 1 and b.human.pending_seq == 1)

    refusal = registry.can_start()

    assert refusal is not None and "already running 2" in refusal
    await registry.shutdown()


# --- failures never take the process with them --------------------------------


async def test_a_run_that_raises_says_so_in_the_thread(mocker):
    graph = mocker.Mock()
    graph.ainvoke = mocker.AsyncMock(side_effect=RuntimeError("bronto is down"))
    registry = _registry(graph)
    api = FakeSlackApi()

    await _start(registry, api).task

    assert any("failed" in p["text"] for p in api.posts)
    assert registry.get("C1", "1.0") is None


async def test_an_unanswered_question_ends_the_run_with_a_notice(mocker):
    """The per-question timeout is the only thing between a distracted human
    and a permanently leaked slot."""
    graph = mock_graph(mocker, _PENDING)
    registry = _registry(graph)
    api = FakeSlackApi()

    await _start(registry, api, answer_timeout=0.01).task

    assert any("stopped waiting" in p["text"] for p in api.posts)
    assert registry.can_start() is None


async def test_a_wedged_run_hits_its_wall_clock(mocker):
    """The per-question timeout does not cover a model call that never returns."""
    graph = mocker.Mock()

    async def never(*args, **kwargs):
        await asyncio.sleep(3600)

    graph.ainvoke = never
    registry = _registry(graph, run_timeout=0.01)
    api = FakeSlackApi()

    await _start(registry, api).task

    assert any("too long" in p["text"] for p in api.posts)


async def test_shutdown_cancels_everything_in_flight(mocker):
    graph = mock_graph(mocker, _PENDING)
    registry = _registry(graph)
    handle = _start(registry, FakeSlackApi())
    await until(lambda: handle.human.pending_seq == 1)

    await registry.shutdown()

    assert handle.task.cancelled()
