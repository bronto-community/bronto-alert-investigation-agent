"""What the user is actually shown, for each of the two questions.

Pure: these build ``Question`` objects and nothing else. Rendering them — as
numbered lines in a terminal, as Block Kit buttons in Slack — is the entrypoint's
job, and the graph never learns which one answered.
"""
# pyright: reportArgumentType=false
# InterruptOnConfig's `description` accepts langchain's private
# _DescriptionFactory protocol, which our plain (tool_call, state, runtime)
# functions satisfy structurally but cannot be declared as.

from typing import Any

from langchain.agents.middleware import InterruptOnConfig
from langchain_core.messages import ToolCall
from langgraph.runtime import Runtime

from ...human import Option, Question
from .confirm import CONFIRM_ALERT, CONFIRM_PLAYBOOK, IdentificationState

RETRY_FRAMING = (
    "The user did not accept any of these candidates. Additional detail: "
    "{text}. Search again with this in mind, then call confirm_alert with "
    "the new candidates."
)


def confirm_alert_question(
    tool_call: ToolCall, state: IdentificationState, runtime: Runtime
) -> dict[str, Any]:
    """Build the confirm-a-candidate Question.

    Returned as a plain dict: it travels in the interrupt's ``description``
    slot, which langchain passes through untouched but which a persistent
    checkpointer would have to serialise.
    """
    candidates: list[dict[str, Any]] = tool_call["args"].get("candidates") or []
    return Question(
        action=CONFIRM_ALERT,
        prompt="Which one are you investigating?",
        options=[
            Option(
                id=str(index),
                label=str(candidate.get("label")),
                subtitle=candidate.get("occurred_at"),
                # Choosing one narrows the proposal to it.
                args={"candidates": [candidate]},
            )
            for index, candidate in enumerate(candidates)
        ],
        free_text="respond",
        respond_template=RETRY_FRAMING,
    ).model_dump(mode="json")


PLAYBOOK_PROMPT = (
    "Optionally provide a playbook or extra context to focus the "
    "investigation: suggested queries to run, and/or dependency services and "
    "logs the agent cannot easily infer."
)


def playbook_question(
    tool_call: ToolCall, state: IdentificationState, runtime: Runtime
) -> dict[str, Any]:
    """Build the playbook Question, offering any pre-configured playbook."""
    configured = state.get("configured_playbook")
    options: list[Option] = []
    prompt = PLAYBOOK_PROMPT
    if configured:
        prompt = "A playbook is already configured for this."
        options = [
            Option(
                id="configured",
                label="Reuse the configured playbook",
                subtitle=configured,
                args={"playbook": configured},
            )
        ]
    return Question(
        action=CONFIRM_PLAYBOOK,
        prompt=prompt,
        options=options,
        free_text="edit",
        text_arg="playbook",
        allow_skip=True,
    ).model_dump(mode="json")


INTERRUPT_ON: dict[str, bool | InterruptOnConfig] = {
    CONFIRM_ALERT: InterruptOnConfig(
        allowed_decisions=["approve", "edit", "respond"],
        description=confirm_alert_question,
    ),
    CONFIRM_PLAYBOOK: InterruptOnConfig(
        allowed_decisions=["approve", "edit"],
        description=playbook_question,
    ),
}
