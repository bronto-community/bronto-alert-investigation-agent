"""The link between an agent and whoever is answering its questions."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


@runtime_checkable
class Human(Protocol):
    """Where questions go and answers come from."""

    async def ask(self, question: Question) -> Answer:
        """Put a question to the human and wait for their answer."""
        ...

    async def tell(self, event: Event) -> None:
        """Show the human something that needs no reply."""
        ...


class Option(BaseModel):
    """One choosable answer, and the tool args choosing it implies."""

    id: str
    label: str
    subtitle: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)


class Question(BaseModel):
    """A structured request for human input, emitted by the graph."""

    action: str
    prompt: str
    options: list[Option] = Field(default_factory=list[Option])
    free_text: Literal["respond", "edit", "none"] = "none"
    text_arg: str | None = None
    respond_template: str = "{text}"
    allow_skip: bool = False

    def option(self, option_id: str) -> Option | None:
        """The option with this id, or None if the question has no such option."""
        return next((o for o in self.options if o.id == option_id), None)

    def validate_answer(self, answer: Answer) -> None:
        """Raise ``UnusableAnswer``, saying why, unless this answer is usable."""
        if answer.skip and not self.allow_skip:
            raise UnusableAnswer(f"{self.action} cannot be skipped")
        if answer.text and self.free_text == "none":
            raise UnusableAnswer(f"{self.action} does not accept free text")
        if answer.option_id is not None and self.option(answer.option_id) is None:
            raise UnusableAnswer(f"there is no option {answer.option_id!r}")
        if answer.is_default and not self.options:
            raise UnusableAnswer("there is nothing to accept")


class UnusableAnswer(ValueError):
    """An answer the question cannot take; the message says why.

    Expected rather than exceptional: a surface catches it, shows the reason,
    and asks again, which costs no checkpoint write and no model call.
    """


class Answer(BaseModel):
    """A human's reply. All three unset means "the default"."""

    option_id: str | None = None
    text: str | None = None
    skip: bool = False

    @property
    def is_default(self) -> bool:
        """Whether this answer accepts the default, by setting nothing."""
        return self.option_id is None and not self.text and not self.skip


class Event(BaseModel):
    """Something a run wants to tell the user that isn't a question."""

    type: Literal["report", "notice"]
    text: str


def question_of(request: dict[str, Any]) -> Question:
    """Recover the structured Question a tool's interrupt config produced."""
    description = request.get("description")
    if isinstance(description, dict):
        return Question.model_validate(description)
    return Question(
        action=request["name"],
        prompt=str(description or request["name"]),
        free_text="respond",
    )


def to_decision(question: Question, answer: Answer) -> dict[str, Any]:
    """Map an answer onto the decision dict the HITL middleware expects."""
    question.validate_answer(answer)

    if answer.skip:
        return _edit(question.action, {_text_arg(question): None})

    if answer.text:
        if question.free_text == "respond":
            # The template is trusted prose from the question; the user's text is
            # only ever a value substituted into it.
            message = question.respond_template.format(text=answer.text)
            return {"type": "respond", "message": message}
        # A free-text answer that edits an arg must not become a `respond`: the
        # tool body would be skipped, the model would re-propose the same call,
        # and the question would be asked again in a loop.
        return _edit(question.action, {_text_arg(question): answer.text})

    return _edit(question.action, _chosen(question, answer).args)


def _text_arg(question: Question) -> str:
    """The arg a skip or a typed value is written to."""
    if question.text_arg is None:
        raise ValueError(f"{question.action} names no text_arg to write the answer to")
    return question.text_arg


def _edit(action: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"type": "edit", "edited_action": {"name": action, "args": args}}


def _chosen(question: Question, answer: Answer) -> Option:
    if answer.option_id is None:
        return question.options[0]  # the default is the model's own proposal
    chosen = question.option(answer.option_id)
    assert chosen is not None  # validate_answer already rejected an unknown id
    return chosen


class NoOneToAsk(RuntimeError):
    """Raised if an unattended run ever reaches a question.

    Unreachable by construction: a run fed an event enters at context
    generation, and nothing downstream of it interrupts. Named rather than
    bare so that, if the graph ever grows a question there, the failure says
    which one.
    """

    def __init__(self, action: str) -> None:
        super().__init__(f"no one to answer {action!r}: this run has no human attached")


class LogSink:
    """Speaks into the log; cannot listen. The report sink with no dependencies.

    What an always-online host gets when no chat workspace is configured: the
    report still leaves the process, into wherever that host collects stdout.
    Raising on ``ask`` rather than answering a default is the point — an
    unattended run that somehow reaches a question should fail loudly instead
    of quietly deciding on someone's behalf.
    """

    def __init__(self, log: logging.Logger | None = None) -> None:
        self._log = log or logger

    async def ask(self, question: Question) -> Answer:
        raise NoOneToAsk(question.action)

    async def tell(self, event: Event) -> None:
        self._log.info("%s:\n%s", event.type, event.text)
