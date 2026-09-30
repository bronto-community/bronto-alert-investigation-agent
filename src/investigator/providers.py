"""Chat-model provider selection and construction."""

import logging
import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Raised when model-provider configuration is missing or invalid."""


SUPPORTED_PROVIDERS = ("anthropic", "bedrock", "openai")

# Map our provider names to LangChain's init_chat_model model_provider values.
_MODEL_PROVIDER = {
    "anthropic": "anthropic",
    "openai": "openai",
    "bedrock": "bedrock_converse",
}

# OpenAI's Chat Completions API makes every tool strict once a response_format
# is set, and strict mode rejects Bronto MCP schemas with optional nested
# fields. The Responses API has no such rule.
_MODEL_KWARGS: dict[str, dict[str, Any]] = {"openai": {"use_responses_api": True}}


# Providers whose configuration is a single required API-key env var.
_SINGLE_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

_TEMPORARY_KEY_PREFIX = "ASIA"


def _validate_bedrock_env() -> None:
    """Bedrock needs a region plus one complete set of credentials."""
    if not (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")):
        raise ConfigError(
            "provider 'bedrock' requires AWS_REGION (or AWS_DEFAULT_REGION) to be set"
        )
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "")
    has_keys = key_id and os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (has_keys or os.environ.get("AWS_PROFILE")):
        raise ConfigError(
            "provider 'bedrock' requires AWS credentials: AWS_ACCESS_KEY_ID and "
            "AWS_SECRET_ACCESS_KEY, or AWS_PROFILE"
        )
    if (
        has_keys
        and key_id.startswith(_TEMPORARY_KEY_PREFIX)
        and not os.environ.get("AWS_SESSION_TOKEN")
    ):
        raise ConfigError(
            f"AWS_ACCESS_KEY_ID starts with {_TEMPORARY_KEY_PREFIX}, so these are "
            "temporary credentials and AWS_SESSION_TOKEN must be set alongside "
            "them. Export all three, or use AWS_PROFILE instead. (An expired "
            "session token fails the same way, from Bedrock rather than here — "
            "`aws sso login` refreshes it.)"
        )


def validate_provider_env(provider: str) -> None:
    """Validate the environment variables required by the selected provider."""
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(SUPPORTED_PROVIDERS)
        raise ConfigError(f"unsupported provider '{provider}' (supported: {supported})")
    if provider == "bedrock":
        _validate_bedrock_env()
        return
    required = _SINGLE_KEY_ENV.get(provider)
    if required and not os.environ.get(required):
        raise ConfigError(f"provider '{provider}' requires {required} to be set")


def build_model(provider: str, model_id: str) -> BaseChatModel:
    """Validate the provider's env, then build the chat model for `model_id`."""
    validate_provider_env(provider)
    logger.info("building chat model provider=%s model=%s", provider, model_id)
    return init_chat_model(
        model_id,
        model_provider=_MODEL_PROVIDER[provider],
        configurable_fields=None,
        config_prefix=None,
        **_MODEL_KWARGS.get(provider, {}),
    )
