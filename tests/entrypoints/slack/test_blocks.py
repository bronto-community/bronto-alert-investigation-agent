"""Tests for rendering a Question into Block Kit."""

import json

import pytest

from investigator.entrypoints.slack.blocks import (
    MARKDOWN_LIMIT,
    MAX_BUTTONS,
    MAX_SELECT_OPTIONS,
    disabled_blocks,
    format_action_id,
    parse_action_id,
    render_event,
    render_question,
)
from investigator.human import Event, Question
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


def _playbook(configured=None) -> Question:
    return Question.model_validate(
        playbook_question({}, {"configured_playbook": configured}, None)
    )


def _many(count: int) -> Question:
    return _alert_question(
        *[{"id": f"m-{i}", "label": f"alert {i}"} for i in range(count)]
    )


def _of_type(blocks, block_type):
    return [b for b in blocks if b["type"] == block_type]


# --- action ids ---------------------------------------------------------------


def test_action_ids_round_trip():
    assert parse_action_id(format_action_id(3, "opt", "0")) == (3, "opt", "0")
    assert parse_action_id(format_action_id(1, "skip")) == (1, "skip", "")


def test_an_option_id_containing_a_colon_survives():
    """The id goes last and is not escaped, so a compound id still parses."""
    assert parse_action_id(format_action_id(2, "opt", "log:42")) == (2, "opt", "log:42")


@pytest.mark.parametrize("raw", ["", "nonsense", "ai:", "ai:x:opt:0", "other:1:opt:0"])
def test_anything_that_is_not_ours_parses_to_none(raw):
    assert parse_action_id(raw) is None


# --- the button path ----------------------------------------------------------


def test_each_candidate_becomes_a_button():
    blocks = render_question(_alert_question(_C0, _C1), seq=1, nonce="n")

    buttons = _of_type(blocks, "actions")[0]["elements"]
    assert [b["text"]["text"] for b in buttons] == [
        "checkout errors",
        "payment latency",
    ]
    assert [b["action_id"] for b in buttons] == ["ai:1:opt:0", "ai:1:opt:1"]


def test_details_go_in_the_summary_because_a_button_has_no_subtitle():
    """The trigger time is what tells two similar alerts apart, and it cannot
    live on the control, so the section carries it."""
    blocks = render_question(_alert_question(_C0, _C1), seq=1, nonce="n")

    section = blocks[0]["text"]["text"]
    assert "*1.* checkout errors — _2026-07-24T14:02:00Z_" in section
    assert "*2.* payment latency — _2026-07-24T09:30:00Z_" in section


def test_a_long_label_is_truncated_to_the_button_limit():
    question = _alert_question({"id": "m-1", "label": "x" * 200})

    button = _of_type(render_question(question, seq=1, nonce="n"), "actions")[0][
        "elements"
    ][0]

    assert len(button["text"]["text"]) == 75
    assert button["text"]["text"].endswith("…")


def test_a_skippable_question_gets_a_skip_button():
    blocks = render_question(_playbook("check redis"), seq=1, nonce="n")

    ids = [e["action_id"] for e in _of_type(blocks, "actions")[0]["elements"]]
    assert ids[-1] == "ai:1:skip"


# --- the select path ----------------------------------------------------------


def test_more_than_five_options_switches_to_a_select():
    """A row of six buttons reads badly long, well before the 25 cap bites."""
    buttons = render_question(_many(MAX_BUTTONS), seq=1, nonce="n")
    select = render_question(_many(MAX_BUTTONS + 1), seq=1, nonce="n")

    assert _of_type(buttons, "actions")[0]["elements"][0]["type"] == "button"
    assert _of_type(select, "actions")[0]["elements"][0]["type"] == "static_select"


def test_the_select_carries_detail_as_an_option_description():
    """Unlike a button, a select's option object does have a description."""
    question = _alert_question(
        *[
            {
                "id": f"m-{i}",
                "label": f"alert {i}",
                "occurred_at": f"2026-07-24T0{i}:00:00Z",
            }
            for i in range(6)
        ]
    )

    select = _of_type(render_question(question, seq=1, nonce="n"), "actions")[0][
        "elements"
    ][0]

    assert select["options"][0]["description"]["text"] == "2026-07-24T00:00:00Z"


def test_the_select_value_carries_the_sequence_so_a_stale_pick_is_refusable():
    select = _of_type(render_question(_many(6), seq=7, nonce="n"), "actions")[0][
        "elements"
    ][0]

    assert select["options"][0]["value"] == "7:0"


def test_more_options_than_a_select_holds_says_so_rather_than_dropping_them():
    """Silently losing candidates would be a correctness bug dressed as a human
    limitation — the whole run's grounding is decided here."""
    blocks = render_question(_many(MAX_SELECT_OPTIONS + 3), seq=1, nonce="n")

    select = _of_type(blocks, "actions")[0]["elements"][0]
    assert len(select["options"]) == MAX_SELECT_OPTIONS
    assert any("3 more not listed" in json.dumps(b) for b in blocks)


# --- the long-detail case that would otherwise 400 ----------------------------


def test_a_long_configured_playbook_gets_its_own_block():
    """playbook_question puts the entire configured playbook in Option.subtitle.
    Inlined into the summary it would blow the section limit and Slack would
    reject the whole message, killing the run at question time."""
    blocks = render_question(_playbook("check redis. " * 400), seq=1, nonce="n")

    assert len(blocks[0]["text"]["text"]) <= 3000
    assert _of_type(blocks, "markdown"), "the playbook needs a block of its own"
    assert "check redis" not in blocks[0]["text"]["text"]


def test_a_short_detail_stays_inline():
    blocks = render_question(_playbook("check redis"), seq=1, nonce="n")

    assert "check redis" in blocks[0]["text"]["text"]
    assert not _of_type(blocks, "markdown")


# --- limits, across every question the graph asks -----------------------------


@pytest.mark.parametrize(
    "question",
    [
        _alert_question(_C0),
        _alert_question(_C0, _C1),
        _many(6),
        _many(MAX_SELECT_OPTIONS + 3),
        _playbook(),
        _playbook("check redis"),
        _playbook("check redis. " * 400),
        _alert_question({"id": "m-1", "label": "x" * 300, "occurred_at": "y" * 300}),
    ],
)
def test_every_rendering_respects_every_block_kit_limit(question):
    blocks = render_question(question, seq=1, nonce="n")

    assert len(blocks) <= 50
    seen_action_ids = []
    for block in blocks:
        assert len(block.get("block_id", "")) <= 255
        if block["type"] == "section":
            assert 1 <= len(block["text"]["text"]) <= 3000
        for element in block.get("elements", []):
            action_id = element.get("action_id")
            if action_id:
                assert len(action_id) <= 255
                seen_action_ids.append(action_id)
            if element.get("type") == "button":
                assert len(element["text"]["text"]) <= 75
            for option in element.get("options", []):
                assert len(option["text"]["text"]) <= 75
                assert len(option["value"]) <= 150
                if "description" in option:
                    assert len(option["description"]["text"]) <= 75
    assert len(seen_action_ids) == len(set(seen_action_ids)), "ids must be distinct"


# --- hints and retirement -----------------------------------------------------


def test_the_hint_says_what_a_thread_reply_would_mean():
    """`respond` sends the agent back to search; `edit` supplies a value."""
    search = render_question(_alert_question(_C0), seq=1, nonce="n")
    own = render_question(_playbook(), seq=1, nonce="n")

    assert "search again" in json.dumps(search)
    assert "with your own" in json.dumps(own)


def test_a_retired_question_has_no_controls_left():
    """Slack has no flag that disables a button, so the block has to go."""
    blocks = disabled_blocks(_alert_question(_C0, _C1), note="_expired_")

    assert not _of_type(blocks, "actions")
    assert "expired" in json.dumps(blocks)


# --- reports ------------------------------------------------------------------


def test_a_short_report_is_one_markdown_block():
    """A markdown block, not a section: the report is LLM Markdown, and mrkdwn
    would render `**bold**` and `## heading` literally."""
    messages = render_event(Event(type="report", text="## Root cause\n\n**GC pause**"))

    assert len(messages) == 1
    assert messages[0][0][0]["type"] == "markdown"
    assert messages[0][0][0]["text"] == "## Root cause\n\n**GC pause**"


def test_a_long_report_splits_across_messages_on_paragraph_boundaries():
    report = "\n\n".join(f"## Section {i}\n\n" + ("word " * 400) for i in range(12))

    messages = render_event(Event(type="report", text=report))

    assert len(messages) > 1
    for blocks, _ in messages:
        assert len(blocks[0]["text"]) <= MARKDOWN_LIMIT
    # Nothing was lost and nothing was split mid-paragraph.
    assert "".join(b[0]["text"] for b, _ in messages).replace(
        "\n\n", ""
    ) == report.replace("\n\n", "")


def test_an_oversized_code_fence_stays_balanced_across_the_split():
    """A hard cut inside a fence would render the tail as prose with stray
    backticks, so the fence is closed and reopened across the seam."""
    messages = render_event(Event(type="report", text="```\n" + "x" * 30_000 + "\n```"))

    assert len(messages) > 1
    for blocks, _ in messages:
        assert blocks[0]["text"].count("```") % 2 == 0


def test_the_fallback_text_is_a_short_notification_not_the_report():
    """`text` is what shows in a push notification; the body belongs in blocks."""
    report = "# Incident report\n\n" + ("word " * 5000)

    _, fallback = render_event(Event(type="report", text=report))[0]

    assert len(fallback) <= 150
    assert fallback.startswith("Incident report")


def test_a_notice_is_one_context_line():
    messages = render_event(Event(type="notice", text="working on it"))

    assert len(messages) == 1
    assert messages[0][0][0]["type"] == "context"
