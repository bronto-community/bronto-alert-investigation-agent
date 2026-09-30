"""Tests for the two questions phase 1 asks, and how they are wired in."""

from investigator.human import Question
from investigator.phases.identify.confirm import (
    CONFIRM_ALERT,
    CONFIRM_PLAYBOOK,
    Candidate,
)
from investigator.phases.identify.questions import (
    INTERRUPT_ON,
    RETRY_FRAMING,
    confirm_alert_question,
    playbook_question,
)

CANDIDATE = Candidate(
    id="mon-1", label="foo bar baz", occurred_at="2026-07-24T14:19:09Z"
)


# --- prompt rendering ---------------------------------------------------------


def test_confirm_alert_question_offers_one_option_per_candidate():
    """The user picks an option, so each candidate becomes one."""
    call = {
        "args": {
            "candidates": [
                CANDIDATE.model_dump(),
                {
                    "id": "mon-2",
                    "label": "latency",
                    "occurred_at": "2026-07-24T09:30:00Z",
                },
            ]
        }
    }

    question = Question.model_validate(confirm_alert_question(call, {}, None))

    assert [(o.id, o.label) for o in question.options] == [
        ("0", "foo bar baz"),
        ("1", "latency"),
    ]
    # Choosing an option narrows the model's proposal down to that candidate.
    assert question.options[1].args["candidates"][0]["id"] == "mon-2"


def test_playbook_question_offers_a_configured_playbook_for_reuse():
    """A pre-configured playbook becomes an option, so the user can accept it."""
    question = Question.model_validate(
        playbook_question({}, {"configured_playbook": "check redis"}, None)
    )

    assert [o.subtitle for o in question.options] == ["check redis"]
    assert question.options[0].args == {"playbook": "check redis"}


def test_playbook_question_without_one_offers_nothing_to_reuse():
    """With nothing configured there is nothing to accept, so don't offer it."""
    question = Question.model_validate(
        playbook_question({}, {"configured_playbook": None}, None)
    )

    assert question.options == []
    assert question.allow_skip


def test_playbook_free_text_edits_rather_than_responds():
    """A `respond` here would skip the tool body, route back to the model, and
    have it re-propose the same call — asking forever."""
    question = Question.model_validate(playbook_question({}, {}, None))

    assert question.free_text == "edit"
    assert question.text_arg == "playbook"


def test_confirm_alert_question_frames_free_text_as_the_rejection_it_is():
    """`respond` produces a success ToolMessage, which on its own reads to the
    model as agreement."""
    question = Question.model_validate(
        confirm_alert_question({"args": {"candidates": []}}, {}, None)
    )

    assert question.free_text == "respond"
    assert question.respond_template == RETRY_FRAMING
    assert "did not accept" in question.respond_template


# --- interrupt configuration --------------------------------------------------


def test_only_the_confirm_alert_question_accepts_a_retry():
    """`respond` skips the tool body and routes back to the model. That is the
    retry path for the triggered alert question, but on the playbook question the model
    would re-propose the same call and interrupt forever. Without
    allowed_decisions at all, the entry is silently dropped."""
    assert INTERRUPT_ON[CONFIRM_ALERT]["allowed_decisions"] == [
        "approve",
        "edit",
        "respond",
    ]
    assert INTERRUPT_ON[CONFIRM_PLAYBOOK]["allowed_decisions"] == ["approve", "edit"]
