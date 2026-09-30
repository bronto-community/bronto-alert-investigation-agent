"""Rendering a Question into Block Kit.

Every function here is pure: a Question in, Block Kit dicts out. Nothing waits,
nothing holds state, and nothing imports the Slack SDK — a test enforces that
last one. Answering a rendered question is ``thread.py``'s job.

The limits below are Block Kit's, not ours, and violating one is a runtime
``invalid_blocks`` error at question time that kills the run — so each is a
named constant with the reference it came from.
"""

from typing import Any

from ...human import Event, Option, Question

# https://docs.slack.dev/reference/block-kit/block-elements/button-element
BUTTON_TEXT = 75
# https://docs.slack.dev/reference/block-kit/blocks/section-block
SECTION_TEXT = 3000
BLOCK_ID = 255
# https://docs.slack.dev/reference/block-kit/composition-objects/option-object
OPTION_TEXT = 75
OPTION_VALUE = 150
OPTION_DESCRIPTION = 75
# https://docs.slack.dev/reference/block-kit/block-elements/select-menu-element
MAX_SELECT_OPTIONS = 100
# https://docs.slack.dev/reference/block-kit/blocks/actions-block — 25 is the
# hard cap, but a row of more than a handful of buttons reads badly long, so we
# switch to a select well before it.
MAX_BUTTONS = 5
# https://docs.slack.dev/reference/block-kit/blocks/markdown-block — the limit
# is cumulative across every markdown block in one payload; we send one per
# message and leave headroom.
MARKDOWN_LIMIT = 11_000
# The notification text, not the body. Kept short deliberately.
FALLBACK_TEXT = 150

# An Option.subtitle longer than this gets its own block rather than joining the
# summary list. The playbook question puts an entire configured playbook here,
# which would otherwise blow SECTION_TEXT and have Slack reject the message.
LONG_DETAIL = 200

_PREFIX = "ai"


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` characters, marking that it was cut."""
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# --- action ids ---------------------------------------------------------------


def format_action_id(seq: int, control: str, option_id: str = "") -> str:
    """Encode which question, and which answer to it, a control stands for.

    ``seq`` is the surface's question counter. It is what makes a click on a
    superseded message refusable rather than an answer to whatever is being
    asked now — the failure mode a chat surface has and a terminal does not.

    The option id goes last and is not escaped, so an id containing a colon
    still round-trips. A control with no option (Skip) omits it entirely.

    Not truncated, unlike ``block_id``: option ids are list indices, so the
    longest id this can build is a couple of dozen characters against Slack's
    limit of 255.
    """
    tail = f":{option_id}" if option_id else ""
    return f"{_PREFIX}:{seq}:{control}{tail}"


def parse_action_id(raw: str) -> tuple[int, str, str] | None:
    """Decode an action id, or None if it is not one of ours."""
    parts = (raw or "").split(":", 3)
    if len(parts) < 3 or parts[0] != _PREFIX or not parts[1].isdigit():
        return None
    return int(parts[1]), parts[2], parts[3] if len(parts) == 4 else ""


# --- blocks -------------------------------------------------------------------


def _section(text: str) -> dict[str, Any]:
    # A section's text has a minimum length of 1; an empty one is a 400.
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": truncate(text, SECTION_TEXT) or "…"},
    }


def context_block(text: str) -> dict[str, Any]:
    """A small grey line of mrkdwn, for notes beneath a message."""
    return {
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": truncate(text, SECTION_TEXT)}],
    }


def _markdown(text: str) -> dict[str, Any]:
    """A markdown block — standard Markdown, unlike mrkdwn.

    This is what lets the report phase stay surface-agnostic: it emits ordinary
    LLM Markdown (``## headings``, ``**bold**``, tables, fenced code) and Slack
    translates it. A section with mrkdwn would render the asterisks literally.
    """
    return {"type": "markdown", "text": text}


def _subtitle_is_long(option: Option) -> bool:
    return bool(option.subtitle) and len(option.subtitle) > LONG_DETAIL


def fallback_text(text: str) -> str:
    """The notification line. Never the body — this is what shows in a push."""
    return truncate(" ".join((text or "").split()), FALLBACK_TEXT)


def _summary_lines(question: Question) -> list[str]:
    """The numbered option list shown above the controls.

    Buttons have no subtitle, so a candidate's ``detail`` — the trigger time
    that distinguishes two similarly-named candidates — has nowhere to go on the
    control itself. It goes here instead, and the buttons are the picker.
    """
    lines: list[str] = []
    for index, option in enumerate(question.options, start=1):
        detail = "" if _subtitle_is_long(option) else (option.subtitle or "")
        suffix = f" — _{detail}_" if detail else ""
        lines.append(f"*{index}.* {option.label}{suffix}")
    return lines


def _buttons(question: Question, seq: int) -> list[dict[str, Any]]:
    elements = [
        {
            "type": "button",
            "text": {"type": "plain_text", "text": truncate(o.label, BUTTON_TEXT)},
            "action_id": format_action_id(seq, "opt", o.id),
        }
        for o in question.options
    ]
    if question.allow_skip:
        elements.append(_skip_button(seq))
    return elements


def _skip_button(seq: int) -> dict[str, Any]:
    return {
        "type": "button",
        "text": {"type": "plain_text", "text": "Skip"},
        "action_id": format_action_id(seq, "skip"),
    }


def _select(question: Question, seq: int) -> dict[str, Any]:
    """One select instead of a long row of buttons.

    The select's option objects do have a ``description``, so this is the one
    path where a short detail sits on the control itself.
    """
    options = []
    for option in question.options[:MAX_SELECT_OPTIONS]:
        entry = {
            "text": {"type": "plain_text", "text": truncate(option.label, OPTION_TEXT)},
            "value": truncate(f"{seq}:{option.id}", OPTION_VALUE),
        }
        if option.subtitle and not _subtitle_is_long(option):
            entry["description"] = {
                "type": "plain_text",
                "text": truncate(option.subtitle, OPTION_DESCRIPTION),
            }
        options.append(entry)
    return {
        "type": "static_select",
        "placeholder": {"type": "plain_text", "text": "Choose one"},
        "options": options,
        "action_id": format_action_id(seq, "select"),
    }


_HINT = {
    "respond": "…or reply in this thread to search again",
    "edit": "…or reply in this thread with your own",
}


def render_question(
    question: Question, *, seq: int, nonce: str
) -> list[dict[str, Any]]:
    """Render a Question as Block Kit."""
    blocks: list[dict[str, Any]] = [
        _section("\n".join([f"*{question.prompt}*", *_summary_lines(question)]))
    ]

    # A detail too long for the summary list gets its own block. Otherwise the
    # playbook question, which carries a whole configured playbook here, would
    # overflow the section and Slack would reject the message.
    for option in question.options:
        if _subtitle_is_long(option):
            blocks.append(_markdown(truncate(option.subtitle or "", SECTION_TEXT)))

    controls: list[dict[str, Any]] = []
    if len(question.options) > MAX_BUTTONS:
        controls.append(_select(question, seq))
        if question.allow_skip:
            controls.append(_skip_button(seq))
        if len(question.options) > MAX_SELECT_OPTIONS:
            dropped = len(question.options) - MAX_SELECT_OPTIONS
            # Said out loud: silently dropping candidates would be a
            # correctness bug dressed as a human limitation.
            blocks.append(
                context_block(
                    f"_showing the first {MAX_SELECT_OPTIONS};"
                    f" {dropped} more not listed_"
                )
            )
    elif question.options or question.allow_skip:
        controls = _buttons(question, seq)

    if controls:
        blocks.append(
            {
                "type": "actions",
                # Unique per message AND per iteration of a message, since an
                # expired question is updated in place.
                "block_id": truncate(f"{_PREFIX}:{nonce}:{seq}", BLOCK_ID),
                "elements": controls,
            }
        )

    hint = _HINT.get(question.free_text)
    if hint:
        blocks.append(context_block(hint))
    return blocks


def disabled_blocks(question: Question, *, note: str) -> list[dict[str, Any]]:
    """The question with its controls removed, for a timed-out message.

    Slack has no flag that disables a button, so the actions block has to go.
    """
    return [_section(f"*{question.prompt}*"), context_block(note)]


# --- events -------------------------------------------------------------------


def _split_paragraphs(text: str, budget: int) -> list[str]:
    """Split on blank lines, packing paragraphs up to ``budget``.

    A paragraph longer than the budget on its own is hard-cut, reopening a
    fenced code block across the seam so the halves still render as code
    rather than as prose with stray backticks.
    """
    chunks: list[str] = []
    current = ""
    for paragraph in (text or "").split("\n\n"):
        if len(paragraph) > budget:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(paragraph, budget))
            continue
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > budget:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks or [""]


def _hard_split(paragraph: str, budget: int) -> list[str]:
    """Cut one oversized paragraph, keeping any code fence balanced."""
    pieces = [paragraph[i : i + budget] for i in range(0, len(paragraph), budget)]
    out: list[str] = []
    open_fence = False
    for piece in pieces:
        prefix = "```\n" if open_fence else ""
        open_fence = (prefix + piece).count("```") % 2 == 1
        suffix = "\n```" if open_fence else ""
        out.append(prefix + piece + suffix)
    return out


def render_event(event: Event) -> list[tuple[list[dict[str, Any]], str]]:
    """Render an Event as (blocks, fallback text) per message to post.

    A report is LLM Markdown and can be long, so it goes into markdown blocks
    split across as many messages as it takes. A notice is one line.
    """
    if event.type == "notice":
        return [([context_block(event.text)], fallback_text(event.text))]
    fallback = fallback_text(
        event.text.lstrip("# ").splitlines()[0] if event.text else "Report"
    )
    return [
        ([_markdown(chunk)], fallback)
        for chunk in _split_paragraphs(event.text, MARKDOWN_LIMIT)
    ]
