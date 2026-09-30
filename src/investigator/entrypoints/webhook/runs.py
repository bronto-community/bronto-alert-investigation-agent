"""Investigations started by a webhook: one asyncio task each, with a limit."""

import asyncio
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from ...human import Event, Human
from ...run import run_config, run_to_completion
from ...state import InvestigationState, evaluation_start, state_for_event

logger = logging.getLogger(__name__)

# Concurrent investigations on one event loop sharing one MCP session. The
# same number, and the same reasoning, as the Slack surface: a hard cap with an
# honest refusal beats a queue a restart would silently drop.
MAX_LIVE_RUNS = 3

# Wall clock for a whole run, so a wedged model call cannot hold a slot for
# ever. There is no per-question timeout here because there are no questions.
RUN_TIMEOUT = 3600.0


class AtCapacity(RuntimeError):
    """Raised when every run slot is taken. The caller answers 503."""


def alert_key(payload: dict[str, Any]) -> str:
    """What makes two payloads the same alert: which monitor, which evaluation."""
    return f"{payload['monitor_id']}@{evaluation_start(payload)}"


def thread_id_for(payload: dict[str, Any]) -> str:
    """The checkpointer's key for one alert."""
    return f"webhook:{alert_key(payload)}"


class EventRuns:
    """Which investigations are in flight, and how to start another."""

    def __init__(
        self,
        graph: CompiledStateGraph[InvestigationState],
        *,
        human_for: Callable[[], Human],
        max_live: int = MAX_LIVE_RUNS,
        run_timeout: float = RUN_TIMEOUT,
    ) -> None:
        self._graph = graph
        self._human_for = human_for
        self._max_live = max_live
        self._run_timeout = run_timeout
        self._live: set[asyncio.Task[None]] = set()

    @property
    def live(self) -> int:
        """How many runs are in flight."""
        return len(self._live)

    def start(self, payload: dict[str, Any]) -> str:
        """Spawn a run for one alert and return its id, or raise ``AtCapacity``.

        Synchronous, so the slot is taken before the caller can await anything
        — two webhooks arriving together cannot both see the last slot free.
        """
        if len(self._live) >= self._max_live:
            raise AtCapacity(f"already running {self._max_live} investigations")
        thread_id = thread_id_for(payload)
        task = asyncio.create_task(
            self._run(payload, thread_id), name=f"run {thread_id}"
        )
        self._live.add(task)
        # create_task results are dropped on the floor by default, so a run
        # that dies takes its traceback with it unless something is listening.
        task.add_done_callback(self._finish)
        return thread_id

    async def shutdown(self) -> None:
        """Cancel every live run and wait for them to unwind."""
        tasks = list(self._live)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._live.clear()

    # -- internals -------------------------------------------------------------

    async def _run(self, payload: dict[str, Any], thread_id: str) -> None:
        human = self._human_for()
        try:
            async with asyncio.timeout(self._run_timeout):
                await run_to_completion(
                    self._graph,
                    state_for_event(payload),
                    human,
                    config=run_config(thread_id),
                )
        except TimeoutError:
            logger.warning("run %s exceeded its wall clock", thread_id)
            await self._notice(human, f"{thread_id}: this investigation took too long.")
        except Exception:
            logger.exception("run %s failed", thread_id)
            await self._notice(
                human,
                f"{thread_id}: the investigation failed. The details are in the logs.",
            )

    def _finish(self, task: asyncio.Task[None]) -> None:
        self._live.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "run ended in an unhandled exception", exc_info=task.exception()
            )

    async def _notice(self, human: Human, text: str) -> None:
        """Say what went wrong; never let saying it be the failure."""
        with contextlib.suppress(Exception):
            await human.tell(Event(type="notice", text=text))
