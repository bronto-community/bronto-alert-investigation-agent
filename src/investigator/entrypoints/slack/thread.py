"""Answering a question that arrives from somewhere else entirely.

A terminal blocks on stdin; Slack does not. A question is posted, the run waits
on a Future, and a click or a thread reply minutes later resolves it — possibly
on a different task, possibly after the question has been superseded. That gap
is what everything here exists to manage.

Rendering lives next door in ``blocks.py`` and knows nothing about waiting.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ...human import Answer, Event, Question, UnusableAnswer
from .blocks import (
    context_block,
    disabled_blocks,
    fallback_text,
    render_event,
    render_question,
)

logger = logging.getLogger(__name__)


class SlackApi(Protocol):
    """The two Slack calls a run makes. Implemented over AsyncWebClient."""

    async def post(
        self, *, blocks: list[dict[str, Any]], text: str, thread_ts: str | None = None
    ) -> str:
        """Post a message, in a thread if ``thread_ts`` is set; return its ts."""
        ...

    async def update(self, *, ts: str, blocks: list[dict[str, Any]], text: str) -> None:
        """Replace the message at ``ts``."""
        ...


class QuestionTimeout(Exception):
    """Nobody answered a question in time; the run is abandoned."""


DEFAULT_ANSWER_TIMEOUT = 900.0  # 15 minutes


@dataclass
class _Pending:
    question: Question
    seq: int
    future: "asyncio.Future[Answer]"
    message_ts: str | None = None


class SlackHuman:
    """Asks in a Slack thread; the answer arrives much later, from a listener.

    ``ask`` blocks on a Future that the bolt app resolves through ``submit``.
    The graph never learns it waited: ``run_to_completion``'s interrupt loop and the
    ``Human`` protocol are untouched, which is the whole point of the seam.

    Unlike the terminal, awaiting here does not block the event loop, so the
    shared MCP session's reader keeps running while a human thinks.
    """

    def __init__(
        self,
        api: SlackApi,
        *,
        thread_ts: str,
        nonce: str,
        answer_timeout: float = DEFAULT_ANSWER_TIMEOUT,
    ) -> None:
        self._api = api
        self._thread_ts = thread_ts
        self._nonce = nonce
        self._timeout = answer_timeout
        self._seq = 0
        self._pending: _Pending | None = None

    # -- Human ---------------------------------------------------------------

    async def ask(self, question: Question) -> Answer:
        while True:
            answer = await self._ask_once(question)
            try:
                question.validate_answer(answer)
            except UnusableAnswer as problem:
                # Re-asked here rather than by resuming the graph: no checkpoint
                # write, no superstep, no model call. The terminal's contract.
                await self._api.post(
                    thread_ts=self._thread_ts,
                    blocks=[context_block(f":warning: {problem}")],
                    text=fallback_text(str(problem)),
                )
                continue
            return answer

    async def tell(self, event: Event) -> None:
        # Sequential rather than gathered: chat.postMessage is rate-limited to
        # roughly one message per second per channel.
        for blocks, text in render_event(event):
            await self._api.post(thread_ts=self._thread_ts, blocks=blocks, text=text)

    # -- answering -------------------------------------------------------------

    def submit(
        self,
        *,
        seq: int,
        option_id: str | None = None,
        text: str | None = None,
        skip: bool = False,
    ) -> str | None:
        """Resolve the pending question. None on success, else why it was refused.

        Does no I/O and no validation — validation belongs to
        ``question.validate_answer`` inside ``ask``, so both user interfaces re-ask for
        the same reasons. That makes this a pure state transition, which is why
        it can be tested without a clock.

        A ``seq`` that is not the pending one is a click on a superseded
        message; refusing it is what stops a stale button answering the
        question currently on screen.
        """
        pending = self._pending
        if pending is None:
            return "there is no question waiting in this thread"
        if seq != pending.seq or pending.future.done():
            return "that question has already been answered"
        pending.future.set_result(Answer(option_id=option_id, text=text, skip=skip))
        return None

    @property
    def pending_seq(self) -> int | None:
        """The sequence number of the question awaiting an answer, if any."""
        return self._pending.seq if self._pending else None

    @property
    def accepts_text(self) -> bool:
        """Whether a thread reply right now is an answer rather than chatter.

        Without this, someone typing "thanks" under a buttons-only question
        would resolve the Future, fail validation, and get the whole question
        re-posted at them.
        """
        return self._pending is not None and self._pending.question.free_text != "none"

    # -- internals -------------------------------------------------------------

    async def _ask_once(self, question: Question) -> Answer:
        self._seq += 1
        pending = _Pending(
            question=question,
            seq=self._seq,
            future=asyncio.get_running_loop().create_future(),
        )
        # Registered before the post: a click can arrive between Slack
        # accepting the message and chat.postMessage returning.
        self._pending = pending
        try:
            pending.message_ts = await self._api.post(
                thread_ts=self._thread_ts,
                blocks=render_question(question, seq=pending.seq, nonce=self._nonce),
                text=fallback_text(question.prompt),
            )
            return await asyncio.wait_for(pending.future, self._timeout)
        except TimeoutError:
            await self._expire(pending)
            raise QuestionTimeout(question.action) from None
        finally:
            if self._pending is pending:
                self._pending = None

    async def _expire(self, pending: _Pending) -> None:
        """Retire a timed-out question so its buttons cannot still be clicked."""
        if pending.message_ts is None:
            return
        note = "_no answer in time — this question expired_"
        try:
            await self._api.update(
                ts=pending.message_ts,
                blocks=disabled_blocks(pending.question, note=note),
                text=fallback_text("question expired"),
            )
        except Exception:  # the run is ending either way
            logger.warning("could not retire the expired question", exc_info=True)
