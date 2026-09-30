"""Configuration: what the app is built from."""

from dataclasses import dataclass

from bronto_sdk import BrontoClient
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

# Model calls the investigation loop may spend. The last one is spent on the
# conclusion rather than another search, so this is the total for the phase.
# Only that loop is bounded by it: identification has its own mechanism, and
# context and report generation each make at most one call outside it.
DEFAULT_MAX_MODEL_CALLS = 5


@dataclass(frozen=True)
class Settings:
    """Everything the process needs to build and run an app.

    Assembled once at the adapter edge from flags and environment; nothing
    below the adapter reads either.

    Build-time only. A per-run value — the event payload, the answer someone
    typed — is passed as an argument instead, so nothing here changes between
    two runs served by the same compiled app.
    """

    provider: str
    model_id: str
    max_model_calls: int = DEFAULT_MAX_MODEL_CALLS
    log_level: str = "INFO"


@dataclass(frozen=True)
class Deps:
    """The dependencies the agent's phases are built from."""

    client: BrontoClient  # the Bronto REST API
    model: BaseChatModel
    investigation_tools: list[BaseTool]
