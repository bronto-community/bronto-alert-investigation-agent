"""Tests for chat-model provider validation and construction."""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableBinding
from langchain_core.tools import StructuredTool

from investigator.phases.investigate import Findings
from investigator.providers import (
    ConfigError,
    build_model,
    can_force_tool_choice,
    validate_provider_env,
)


def test_anthropic_missing_api_key_raises(monkeypatch):
    """Choosing anthropic without ANTHROPIC_API_KEY fails, naming var + provider."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc:
        validate_provider_env("anthropic")

    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "anthropic" in message


def test_anthropic_with_key_validates(monkeypatch):
    """Anthropic validates when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-xxx")

    validate_provider_env("anthropic")  # no raise


def test_openai_missing_api_key_raises(monkeypatch):
    """Selecting openai without OPENAI_API_KEY fails, naming the var + provider."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ConfigError) as exc:
        validate_provider_env("openai")

    message = str(exc.value)
    assert "OPENAI_API_KEY" in message
    assert "openai" in message


def test_openai_with_key_validates(monkeypatch):
    """OpenAI validates when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")

    validate_provider_env("openai")  # no raise


def test_bedrock_region_and_profile_validates(monkeypatch):
    """Bedrock validates with AWS_REGION and AWS_PROFILE (no access keys)."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_PROFILE", "sandbox")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)

    validate_provider_env("bedrock")  # no raise


def test_bedrock_missing_region_raises(monkeypatch):
    """Bedrock requires a region even when credentials are present."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    with pytest.raises(ConfigError) as exc:
        validate_provider_env("bedrock")

    assert "AWS_REGION" in str(exc.value)


def test_bedrock_region_but_no_credentials_raises(monkeypatch):
    """Bedrock with a region but neither keys nor profile fails, naming the creds."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    with pytest.raises(ConfigError) as exc:
        validate_provider_env("bedrock")

    message = str(exc.value)
    assert "AWS_ACCESS_KEY_ID" in message
    assert "AWS_PROFILE" in message


def test_bedrock_partial_keys_no_profile_raises(monkeypatch):
    """Bedrock with only one of the two access-key vars (and no profile) fails."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    with pytest.raises(ConfigError, match="AWS_SECRET_ACCESS_KEY, or AWS_PROFILE"):
        validate_provider_env("bedrock")


def test_bedrock_default_region_alone_validates(monkeypatch):
    """AWS_DEFAULT_REGION is a region too — it is what boto3 falls back to."""
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    validate_provider_env("bedrock")  # no raise


def test_bedrock_temporary_keys_without_a_session_token_raises(monkeypatch):
    """An ASIA key pair alone is the failure this check exists for.

    Exporting only the two key vars out of an SSO profile reaches Bedrock and
    comes back `UnrecognizedClientException`, from inside the first model call.
    """
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ASIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    with pytest.raises(ConfigError) as exc:
        validate_provider_env("bedrock")

    assert "AWS_SESSION_TOKEN" in str(exc.value)


def test_bedrock_temporary_keys_with_a_session_token_validates(monkeypatch):
    """The same keys with the third variable are a complete credential."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "ASIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "FwoGZXIvYXdzE...")
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    validate_provider_env("bedrock")  # no raise


def test_bedrock_long_term_keys_need_no_session_token(monkeypatch):
    """An AKIA pair is long-lived, so the session-token check must not fire."""
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)

    validate_provider_env("bedrock")  # no raise


def test_unknown_provider_raises():
    """An unsupported provider fails, naming the provider and the supported set."""
    with pytest.raises(ConfigError) as exc:
        validate_provider_env("gemini")

    message = str(exc.value)
    assert "gemini" in message
    for provider in ("bedrock", "anthropic", "openai"):
        assert provider in message


def test_build_model_validates_env_before_constructing(monkeypatch, mocker):
    """build_model validates provider env first; on failure it never constructs."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    init = mocker.patch("investigator.providers.init_chat_model")

    with pytest.raises(ConfigError):
        build_model("anthropic", "claude-x")

    init.assert_not_called()


@pytest.mark.parametrize(
    "provider, env, model_provider, responses_api",
    [
        ("anthropic", {"ANTHROPIC_API_KEY": "sk-ant-xxx"}, "anthropic", None),
        ("openai", {"OPENAI_API_KEY": "sk-xxx"}, "openai", True),
        (
            "bedrock",
            {"AWS_REGION": "eu-west-1", "AWS_PROFILE": "sandbox"},
            "bedrock_converse",
            None,
        ),
    ],
)
def test_build_model_maps_each_provider(
    monkeypatch, mocker, provider, env, model_provider, responses_api
):
    """Each provider name maps onto the LangChain model_provider that serves it;
    only OpenAI is put on its Responses API."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    init = mocker.patch("investigator.providers.init_chat_model")

    build_model(provider, "model-x")

    init.assert_called_once()
    assert init.call_args.args == ("model-x",)
    assert init.call_args.kwargs["model_provider"] == model_provider
    assert init.call_args.kwargs.get("use_responses_api") == responses_api


def test_openai_tools_are_not_forced_strict_beside_a_response_format(monkeypatch):
    """Native structured output on Chat Completions forces every tool strict, and
    strict mode rejects an MCP schema with an optional nested field (Bronto's
    `timeseries.delta.natural`), failing the investigation with a 400."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    model = build_model("openai", "gpt-4o-mini")
    timeseries = StructuredTool(
        name="timeseries",
        description="Run a timeseries.",
        args_schema={
            "type": "object",
            "properties": {
                "delta": {
                    "type": "object",
                    "properties": {"natural": {"type": "string"}},
                }
            },
        },
    )

    bound = model.bind_tools([timeseries], response_format=Findings)

    assert isinstance(bound, RunnableBinding)
    [formatted] = bound.kwargs["tools"]
    assert not formatted["function"].get("strict")


@pytest.mark.parametrize(
    "provider, model_id, env",
    [
        ("anthropic", "claude-3-5-haiku-latest", {"ANTHROPIC_API_KEY": "sk-ant-test"}),
        ("openai", "gpt-4o-mini", {"OPENAI_API_KEY": "sk-test"}),
        (
            "bedrock",
            "anthropic.claude-3-haiku-20240307-v1:0",
            {
                "AWS_REGION": "eu-west-1",
                "AWS_ACCESS_KEY_ID": "AKIAEXAMPLE",
                "AWS_SECRET_ACCESS_KEY": "secret",
            },
        ),
    ],
)
def test_build_model_constructs_real_model(provider, model_id, env, monkeypatch):
    """Each provider's SDK is installed and build_model constructs a real model."""
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    model = build_model(provider, model_id)

    assert isinstance(model, BaseChatModel)


@pytest.mark.parametrize(
    "model_id, can_force",
    [
        ("eu.anthropic.claude-sonnet-5", True),
        # Bedrock rejects tool_choice "any" for it: a ValidationException on
        # langchain-aws releases that did not know.
        ("eu.anthropic.claude-opus-5-5", False),
    ],
)
def test_bedrock_reports_whether_a_tool_call_can_be_forced(
    model_id, can_force, monkeypatch
):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")

    assert can_force_tool_choice(build_model("bedrock", model_id)) is can_force


@pytest.mark.parametrize(
    "model_id, can_force",
    [
        ("claude-sonnet-5", True),
        # The Claude API rejects tool_choice "any" and "tool" for these with a
        # 400, and ChatAnthropic does not say so itself.
        ("claude-opus-5-5", False),
        ("claude-sonnet-5-5", False),
        ("claude-fable-5-1", False),
    ],
)
def test_anthropic_reports_whether_a_tool_call_can_be_forced(
    model_id, can_force, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    assert can_force_tool_choice(build_model("anthropic", model_id)) is can_force


def test_openai_can_force_a_tool_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert can_force_tool_choice(build_model("openai", "gpt-5"))
