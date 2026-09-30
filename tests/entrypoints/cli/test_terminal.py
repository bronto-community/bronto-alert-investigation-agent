"""Tests for the terminal human — all the prose and parsing live here now."""

import io

from investigator.entrypoints.cli.terminal import (
    TerminalHuman,
    make_output_fn,
    parse,
    render,
)
from investigator.human import Answer, Event, Question
from investigator.phases.identify.questions import (
    confirm_alert_question,
    playbook_question,
)

_C0 = {"id": "m-1", "label": "checkout errors", "occurred_at": "2026-07-24T14:02:00Z"}
_C1 = {"id": "m-2", "label": "payment latency", "occurred_at": "2026-07-24T09:30:00Z"}


def _alert_question(*candidates) -> Question:
    return Question.model_validate(
        confirm_alert_question({"args": {"candidates": list(candidates)}}, {}, None)
    )


async def _ask(question, *typed):
    answers = iter(typed)
    outputs: list[str] = []
    human = TerminalHuman(input_fn=lambda: next(answers), output_fn=outputs.append)
    return await human.ask(question), outputs


# --- colour -------------------------------------------------------------------


def test_make_output_fn_colors_on_a_tty(mocker):
    stream = mocker.Mock()
    stream.isatty.return_value = True
    mocker.patch("builtins.print")

    make_output_fn(stream)("hi")

    print.assert_called_once_with("\033[1;36mhi\033[0m", file=stream)


def test_make_output_fn_plain_when_not_a_tty():
    stream = io.StringIO()

    make_output_fn(stream)("hi")

    assert stream.getvalue() == "hi\n"


# --- rendering ----------------------------------------------------------------


def test_candidates_are_listed_with_the_ids_the_user_types():
    rendered = render(_alert_question(_C0, _C1))

    assert "Which one are you investigating?" in rendered
    assert "[0] checkout errors" in rendered
    assert "[1] payment latency" in rendered


def test_the_instruction_line_is_built_from_the_questions_own_fields():
    """A new question needs no new prose here."""
    single = render(_alert_question(_C0)).splitlines()[-1]
    several = render(_alert_question(_C0, _C1)).splitlines()[-1]

    assert "press enter to accept [0]" in single
    assert "type an index to pick another" not in single
    assert "type an index to pick another" in several


def test_a_skippable_question_says_so():
    rendered = render(Question.model_validate(playbook_question({}, {}, None)))

    assert "press enter to skip" in rendered


def test_a_question_with_nothing_to_explain_gets_no_instruction_line():
    """The opening 'describe the alert' prompt is the whole instruction."""
    question = Question(action="describe", prompt="Describe it:", free_text="respond")

    assert render(question) == "Describe it:"


# --- parsing ------------------------------------------------------------------


def test_enter_is_the_default_answer():
    assert parse(_alert_question(_C0), "") == Answer()


def test_a_digit_is_an_option_id_only_when_there_are_options():
    assert parse(_alert_question(_C0, _C1), "1") == Answer(option_id="1")
    # Someone whose playbook is literally "1" meant it as text.
    playbook = Question.model_validate(playbook_question({}, {}, None))
    assert parse(playbook, "1") == Answer(text="1")


def test_the_word_skip_only_skips_where_skipping_is_allowed():
    playbook = Question.model_validate(playbook_question({}, {}, None))

    assert parse(playbook, "skip") == Answer(skip=True)
    assert parse(_alert_question(_C0), "skip") == Answer(text="skip")


def test_enter_on_a_question_with_no_options_skips():
    """Nothing was proposed, so accepting the default means going without."""
    playbook = Question.model_validate(playbook_question({}, {}, None))

    assert parse(playbook, "") == Answer(skip=True)


# --- asking -------------------------------------------------------------------


async def test_ask_prints_the_prompt_and_returns_the_answer():
    answer, outputs = await _ask(_alert_question(_C0), "")

    assert "Which one are you investigating?" in outputs[0]
    assert answer == Answer()


async def test_an_unusable_answer_is_reasked_locally():
    """Re-asking here costs no checkpoint write, superstep or model call."""
    answer, outputs = await _ask(_alert_question(_C0, _C1), "5", "1")

    assert any("no option '5'" in out for out in outputs)
    assert answer == Answer(option_id="1")


async def test_tell_prints_the_event_text():
    outputs: list[str] = []
    human = TerminalHuman(input_fn=lambda: "", output_fn=outputs.append)

    await human.tell(Event(type="report", text="the report"))

    assert outputs == ["the report"]
