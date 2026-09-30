"""Report generation: the final, user-facing output."""

import json
import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..state import InvestigationState, alert_context_text

logger = logging.getLogger(__name__)


def make_report_node(model: BaseChatModel):
    """Build the report node.

    One plain model call turns the alert context and findings into a human-readable
    message.
    """

    async def report(state: InvestigationState) -> dict[str, Any]:
        if not state.get("findings"):
            logger.warning(
                "investigation ended without findings; emitting the inconclusive report"
            )
            return {"report": INCONCLUSIVE_REPORT}
        prompt = [
            SystemMessage(REPORT_PROMPT),
            HumanMessage(
                f"Alert context:\n{alert_context_text(state['alert_context'])}\n\n"
                f"Findings:\n{json.dumps(state['findings'], default=str)}"
            ),
        ]
        response = await model.ainvoke(prompt)
        return {"report": response.text}

    return report


REPORT_PROMPT = (
    "Write a clear, human-readable report of the investigation for an "
    "on-call engineer, from the alert context and findings. Cover, in order:\n"
    "1. What the monitor and its triggered event mean, in simple terms.\n"
    "2. The implications of the triggered event, based on the findings.\n"
    "3. The root cause and the evidence supporting it.\n"
    "4. Suggested remediation / next steps.\n"
    "Keep it neither too verbose nor too terse - include all relevant "
    "information and nothing extraneous. Write it as a direct explanation; do "
    "not describe your own confidence in the analysis."
)

# What the user gets when the investigation produced no findings at all. The
# budget's last turn is spent forcing the conclusion, so reaching this means the
# model's final answer was missing or did not parse as findings. Nothing grounded
# is left to report from, so the report says so rather than guessing.
INCONCLUSIVE_REPORT = (
    "The investigation ended without reaching a conclusion — the model did not "
    "return usable findings. The evidence gathered so far is on the trace. "
    "Re-run, or raise --max-model-calls."
)
