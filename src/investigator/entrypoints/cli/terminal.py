"""The terminal surface: every piece of prose and parsing the CLI ever needed.

Nothing above this file knows about indices, the word "skip", ANSI colour, or
stdin. Slack renders the same ``Question`` its own way, in ``slack/blocks.py``.
"""

import asyncio
import sys
from collections.abc import Callable
from typing import TextIO

from ...human import Answer, Event, Question, UnusableAnswer


class TerminalHuman:
    """Asks on stdout, reads from stdin."""

    def __init__(
        self,
        input_fn: Callable[[], str] = input,
        output_fn: Callable[[str], None] | None = None,
    ) -> None:
        self._input = input_fn
        self._output = output_fn or make_output_fn()

    async def ask(self, question: Question) -> Answer:
        while True:
            self._output(render(question))
            answer = parse(question, await self._read())
            try:
                question.validate_answer(answer)
            except UnusableAnswer as problem:
                # Re-asked here rather than by resuming the graph: no checkpoint
                # write, no superstep, no model call.
                self._output(f"! {problem}")
                continue
            return answer

    async def tell(self, event: Event) -> None:
        self._output(event.text)

    async def _read(self) -> str:
        """Read one line without blocking the event loop.

        The MCP session's streamable-HTTP reader shares this loop, so a bare
        blocking ``input()`` would stall it — long enough to trip the server's
        idle timeout while the user is still deciding.
        """
        return await asyncio.to_thread(self._input)


def render(question: Question) -> str:
    """Render a question as the block of text the user is shown."""
    lines = [question.prompt]
    for option in question.options:
        detail = f" — {option.subtitle}" if option.subtitle else ""
        lines.append(f"  [{option.id}] {option.label}{detail}")
    instructions = _instructions(question)
    if instructions:
        lines.append(instructions)
    return "\n".join(lines)


def _instructions(question: Question) -> str:
    """The line telling the user what they may type.

    Assembled from the question's own fields, so a new question needs no new
    prose here.
    """
    parts: list[str] = []
    if question.options:
        parts.append(f"press enter to accept [{question.options[0].id}]")
        if len(question.options) > 1:
            parts.append("type an index to pick another")
    elif question.allow_skip:
        parts.append("press enter to skip")
    if question.free_text == "respond" and question.options:
        # Only a rejection of something proposed reads as "search again"; with
        # nothing proposed the prompt itself is the whole instruction.
        parts.append("describe it again to search once more")
    elif question.free_text == "edit":
        parts.append("type your own")
    if question.allow_skip and question.options:
        parts.append("type 'skip' to go without")
    if not parts:
        return ""
    if len(parts) > 1:
        parts[-1] = f"or {parts[-1]}"
    return f"({', '.join(parts)})"


def parse(question: Question, text: str) -> Answer:
    """Interpret one typed line as an answer to ``question``.

    An index is only an index when the question has options to index into;
    otherwise a digit is just text someone typed as their playbook.
    """
    text = text.strip()
    if not text:
        return Answer(skip=True) if not question.options else Answer()
    if question.allow_skip and text.lower() == "skip":
        return Answer(skip=True)
    if question.options and text.isdigit():
        return Answer(option_id=text)
    return Answer(text=text)


# Bold cyan for user-facing prompts and the report, so they stand out from log
# lines (which logging sends to stderr) when running with verbose logging.
_PROMPT_STYLE = "\033[1;36m"
_RESET = "\033[0m"


def make_output_fn(stream: TextIO = sys.stdout) -> Callable[[str], None]:
    """Build the user-facing output fn.

    It colours when ``stream`` is a TTY and leaves messages plain when piped or
    redirected.
    """
    color = stream.isatty()

    def output(message: str) -> None:
        print(f"{_PROMPT_STYLE}{message}{_RESET}" if color else message, file=stream)

    return output
