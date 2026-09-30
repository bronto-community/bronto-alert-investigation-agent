"""Tests for mapping a structured Answer onto a HITL decision.

There is one translator for every question the graph asks, so these cover
both of them, plus the protocol every surface implements.
"""

import pytest

from investigator.entrypoints.cli.terminal import TerminalHuman
from investigator.entrypoints.slack.post import SlackSink
from investigator.entrypoints.slack.thread import SlackHuman
from investigator.human import (
    Answer,
    Human,
    LogSink,
    Option,
    Question,
    UnusableAnswer,
    to_decision,
)
from investigator.phases.identify.confirm import (
    CONFIRM_ALERT,
    CONFIRM_PLAYBOOK,
)
from investigator.phases.identify.questions import (
    RETRY_FRAMING,
    confirm_alert_question,
    playbook_question,
)

_C0 = {"id": "m-1", "label": "checkout errors", "occurred_at": "2026-07-24T14:02:00Z"}
_C1 = {"id": "m-2", "label": "payment latency", "occurred_at": "2026-07-24T09:30:00Z"}


def _alert_question(*candidates) -> Question:
    return Question.model_validate(
        confirm_alert_question({"args": {"candidates": list(candidates)}}, {}, None)
    )


def _playbook(configured="configured text") -> Question:
    return Question.model_validate(
        playbook_question({}, {"configured_playbook": configured}, None)
    )


def test_default_answer_confirms_a_single_candidate():
    """One candidate and no input means accept the proposal as it stands."""
    assert to_decision(_alert_question(_C0), Answer()) == {
        "type": "edit",
        "edited_action": {"name": CONFIRM_ALERT, "args": {"candidates": [_C0]}},
    }


def test_default_answer_narrows_a_multi_candidate_proposal():
    """The tool body confirms exactly one candidate, so accepting the first of
    several is an edit that narrows the list rather than a bare approve."""
    decision = to_decision(_alert_question(_C0, _C1), Answer())

    assert decision == {
        "type": "edit",
        "edited_action": {"name": CONFIRM_ALERT, "args": {"candidates": [_C0]}},
    }


def test_choosing_an_option_narrows_to_that_candidate():
    decision = to_decision(_alert_question(_C0, _C1), Answer(option_id="1"))

    assert decision["edited_action"]["args"]["candidates"] == [_C1]


def test_an_unknown_option_is_rejected_rather_than_guessed():
    """Every human gets this check from the question, so an unusable answer
    is re-asked locally rather than resumed into the graph."""
    question = _alert_question(_C0, _C1)

    with pytest.raises(UnusableAnswer, match="there is no option '5'"):
        question.validate_answer(Answer(option_id="5"))
    with pytest.raises(ValueError, match="no option"):
        to_decision(question, Answer(option_id="5"))


def test_free_text_sends_the_model_back_to_search():
    """`respond` skips the tool body and routes back to the model. It returns a
    success-status message, so the correction has to be framed as a rejection."""
    decision = to_decision(_alert_question(_C0), Answer(text="try the payment service"))

    assert decision["type"] == "respond"
    assert decision["message"] == RETRY_FRAMING.format(text="try the payment service")


def test_braces_in_free_text_survive_the_framing():
    """The template is trusted prose; the user's text is only ever a value."""
    decision = to_decision(_alert_question(_C0), Answer(text="err {code} at 50%"))

    assert "err {code} at 50%" in decision["message"]


def test_default_answer_reuses_the_configured_playbook():
    """Accepting takes the playbook the monitor already had configured. It must
    be an edit: the model proposed the call before the playbook was fetched, so
    approving its own args would record a null playbook."""
    decision = to_decision(_playbook(), Answer())

    assert decision["edited_action"]["args"] == {"playbook": "configured text"}


def test_own_text_replaces_the_playbook():
    decision = to_decision(_playbook(), Answer(text="my own playbook"))

    assert decision["edited_action"]["args"] == {"playbook": "my own playbook"}


def test_own_text_edits_even_with_nothing_configured():
    """With no option to accept, typing is still an edit, not a respond."""
    decision = to_decision(_playbook(configured=None), Answer(text="mine"))

    assert decision["type"] == "edit"
    assert decision["edited_action"]["name"] == CONFIRM_PLAYBOOK


def test_skip_sets_an_explicit_null_playbook():
    """Skipping must be an edit: a `respond` would skip the tool body, route
    back to the model, and interrupt on the same call forever."""
    decision = to_decision(_playbook(), Answer(skip=True))

    assert decision["type"] == "edit"
    assert decision["edited_action"]["args"] == {"playbook": None}


def test_a_question_that_cannot_be_skipped_rejects_a_skip():
    with pytest.raises(ValueError, match="cannot be skipped"):
        to_decision(_alert_question(_C0), Answer(skip=True))


def test_a_question_that_takes_no_free_text_rejects_text():
    question = Question(action="pick", prompt="?", options=[Option(id="0", label="a")])

    with pytest.raises(ValueError, match="does not accept free text"):
        to_decision(question, Answer(text="anything"))


@pytest.mark.parametrize(
    "surface",
    [TerminalHuman, SlackHuman, SlackSink, LogSink],
    ids=lambda s: s.__name__,
)
def test_every_surface_satisfies_the_protocol(surface):
    """`runtime_checkable` verifies method presence only, but presence is
    exactly what a half-finished rename breaks."""
    assert issubclass(surface, Human)
