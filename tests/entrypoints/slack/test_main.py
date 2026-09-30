"""Tests for the Slack adapter's entry point: arguments and env validation."""

import logging

import pytest

from investigator.entrypoints.slack.main import main, missing_scopes

_ARGS = ["--provider", "anthropic", "--model-id", "claude-x"]


def _env(monkeypatch, **overrides):
    values: dict[str, str | None] = {
        "BRONTO_API_KEY": "secret-key",
        "BRONTO_REGION": "eu",
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
        "SLACK_BOT_TOKEN": "xoxb-x",
        "SLACK_APP_TOKEN": "xapp-x",
    }
    values.update(overrides)
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_it_serves_with_everything_configured(monkeypatch, mocker):
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.slack.main.build_model")
    # Autospec'd so a renamed parameter fails here rather than in a workspace.
    serve = mocker.patch("investigator.entrypoints.slack.main.serve", autospec=True)

    main(_ARGS)

    assert serve.call_args.kwargs["bot_token"] == "xoxb-x"
    assert serve.call_args.kwargs["app_token"] == "xapp-x"
    assert serve.call_args.kwargs["settings"].model_id == "claude-x"


@pytest.mark.parametrize("missing", ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"])
def test_a_missing_slack_token_names_itself(monkeypatch, mocker, capsys, missing):
    _env(monkeypatch, **{missing: None})
    mocker.patch("investigator.entrypoints.slack.main.build_model")
    mocker.patch("investigator.entrypoints.slack.main.serve")

    with pytest.raises(SystemExit) as exc:
        main(_ARGS)

    assert exc.value.code == 1
    assert missing in capsys.readouterr().err


@pytest.mark.parametrize("missing", ["BRONTO_API_KEY", "ANTHROPIC_API_KEY"])
def test_bronto_and_provider_env_are_validated_before_connecting(
    monkeypatch, mocker, capsys, missing
):
    _env(monkeypatch, **{missing: None})
    mocker.patch("investigator.entrypoints.slack.main.build_model")
    serve = mocker.patch("investigator.entrypoints.slack.main.serve")

    with pytest.raises(SystemExit) as exc:
        main(_ARGS)

    assert exc.value.code == 1
    assert missing in capsys.readouterr().err
    serve.assert_not_called()


def test_it_warns_that_runs_do_not_survive_a_restart(monkeypatch, mocker, caplog):
    """Said at startup rather than discovered during an incident."""
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.slack.main.build_model")
    mocker.patch("investigator.entrypoints.slack.main.serve")

    with caplog.at_level(logging.WARNING, logger="investigator.entrypoints.slack.main"):
        main(_ARGS)

    assert "do not survive a restart" in caplog.text


def test_the_concurrency_cap_is_configurable(monkeypatch, mocker):
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.slack.main.build_model")
    serve = mocker.patch("investigator.entrypoints.slack.main.serve", autospec=True)

    main([*_ARGS, "--max-live", "1"])

    assert serve.call_args.kwargs["max_live"] == 1


def test_an_unknown_provider_is_rejected_up_front(monkeypatch, capsys):
    """On the message, not just the exit code — see the CLI twin."""
    _env(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main(["--provider", "nonesuch", "--model-id", "m"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_the_required_scopes_are_all_accounted_for():
    """A token carrying everything the README asks for is missing nothing."""
    granted = "commands,chat:write,app_mentions:read,channels:history,groups:history"

    assert missing_scopes(granted) == []


def test_a_scope_the_token_never_got_is_named():
    """The reason @mentions arrive nowhere, said at startup instead of never."""
    granted = "commands,chat:write,app_mentions:read"

    assert missing_scopes(granted) == ["channels:history", "groups:history"]


def test_an_unreadable_scope_header_claims_nothing_is_missing():
    """Slack not telling us is not evidence that a scope is absent."""
    assert missing_scopes(None) == []
    assert missing_scopes("") == []
