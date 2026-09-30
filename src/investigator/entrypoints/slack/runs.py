"""Live runs: one asyncio task per Slack thread, and the bookkeeping around it.

Nothing here imports Slack. A run is started with an object satisfying
``SlackApi``; the bolt app is what supplies a real one.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph

from ...human import Event
from ...run import DESCRIBE_REQUEST, run_config, run_to_completion
from ...state import InvestigationState, state_for_request
from .thread import DEFAULT_ANSWER_TIMEOUT, QuestionTimeout, SlackApi, SlackHuman

logger = logging.getLogger(__name__)

# Three concurrent investigations on one event loop, sharing one MCP socket, is
# already ambitious. A hard limit with an honest refusal beats a queue with a
# position notice, which is strictly more code for a small demo project.
MAX_LIVE_RUNS = 3

# Wall clock for a whole run, so a wedged model call cannot hold a slot for
# ever. The per-question timeout lives on the SlackHuman.
RUN_TIMEOUT = 3600.0


def thread_id_for(team_id: str, channel_id: str, thread_ts: str) -> str:
    """The checkpointer's key for one Slack thread.

    A ``ts`` is only unique within a channel and the checkpointer is
    process-global, so the channel has to be in the key. The team costs nothing
    and covers a shared-channel install.
    """
    return f"slack:{team_id}:{channel_id}:{thread_ts}"


@dataclass
class RunHandle:
    """One live investigation and the thread it is happening in."""

    key: tuple[str, str]
    thread_id: str
    human: SlackHuman
    request_text: str
    task: "asyncio.Task[None] | None" = field(default=None, repr=False)


class RunRegistry:
    """Which threads have a live run, and how to start another.

    Deliberately not a graph concern: the graph knows about ``thread_id`` and
    nothing else. This is the adapter's own bookkeeping.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[InvestigationState],
        *,
        max_live: int = MAX_LIVE_RUNS,
        run_timeout: float = RUN_TIMEOUT,
    ) -> None:
        self._graph = graph
        self._max_live = max_live
        self._run_timeout = run_timeout
        self._live: dict[tuple[str, str], RunHandle] = {}

    # -- lookup ----------------------------------------------------------------

    def get(self, channel_id: str, thread_ts: str) -> RunHandle | None:
        """The live run in this thread, if there is one."""
        return self._live.get((channel_id, thread_ts))

    def can_start(self) -> str | None:
        """None if a run may start, else the reason it may not."""
        if len(self._live) >= self._max_live:
            return (
                f"I'm already running {self._max_live} investigations. "
                "Try again when one finishes."
            )
        return None

    # -- lifecycle -------------------------------------------------------------

    def start(
        self,
        api: SlackApi,
        *,
        team_id: str,
        channel_id: str,
        thread_ts: str,
        request_text: str,
        answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
    ) -> RunHandle:
        """Spawn a run for this thread. The caller has already checked can_start."""
        key = (channel_id, thread_ts)
        thread_id = thread_id_for(team_id, channel_id, thread_ts)
        human = SlackHuman(
            api,
            thread_ts=thread_ts,
            nonce=thread_ts.replace(".", ""),
            answer_timeout=answer_timeout,
        )
        handle = RunHandle(
            key=key, thread_id=thread_id, human=human, request_text=request_text
        )
        self._live[key] = handle
        handle.task = asyncio.create_task(self._run(handle), name=f"run {thread_id}")
        # create_task results are dropped on the floor by default, so a run that
        # dies takes its traceback with it unless something is listening.
        handle.task.add_done_callback(lambda task: self._finish(handle, task))
        return handle

    async def shutdown(self) -> None:
        """Cancel every live run and wait for them to unwind."""
        tasks = [h.task for h in self._live.values() if h.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._live.clear()

    # -- internals -------------------------------------------------------------

    async def _run(self, handle: RunHandle) -> None:
        human = handle.human
        try:
            async with asyncio.timeout(self._run_timeout):
                text = handle.request_text
                if not text:
                    # No description in the trigger, so ask for one — the same
                    # Question the CLI asks, through the same human.
                    text = (await human.ask(DESCRIBE_REQUEST)).text or ""
                await run_to_completion(
                    self._graph,
                    state_for_request(text),
                    human,
                    config=run_config(handle.thread_id),
                )
        except QuestionTimeout:
            await self._notice(
                human,
                ":alarm_clock: No answer, so I've stopped waiting. "
                "Start again with `/investigate` when you're ready.",
            )
        except TimeoutError:
            logger.warning("run %s exceeded its wall clock", handle.thread_id)
            await self._notice(human, ":alarm_clock: This investigation took too long.")
        except Exception:
            logger.exception("run %s failed", handle.thread_id)
            await self._notice(
                human, ":x: The investigation failed. The details are in the logs."
            )

    def _finish(self, handle: RunHandle, task: "asyncio.Task[None]") -> None:
        self._live.pop(handle.key, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error(
                "run %s ended in an unhandled exception",
                handle.thread_id,
                exc_info=task.exception(),
            )

    async def _notice(self, human: SlackHuman, text: str) -> None:
        """Tell the thread something went wrong; never let that be the failure."""
        with contextlib.suppress(Exception):
            await human.tell(Event(type="notice", text=text))
