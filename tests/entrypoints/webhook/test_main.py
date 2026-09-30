"""Tests for the webhook surface's entry point: arguments, env, report sink."""

import re
from pathlib import Path

import pytest

from investigator.entrypoints import webhook
from investigator.entrypoints.webhook.main import main, report_sink
from investigator.human import LogSink
from investigator.providers import ConfigError

_ARGS = ["--provider", "anthropic", "--model-id", "claude-x"]


def _env(monkeypatch, **overrides):
    values: dict[str, str | None] = {
        "BRONTO_API_KEY": "secret-key",
        "BRONTO_REGION": "eu",
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
        "BRONTO_WEBHOOK_SECRET": "shh",
    }
    values.update(overrides)
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


# --- where the report goes ----------------------------------------------------


def test_logs_is_the_sink_that_needs_nothing():
    """The default, so an always-online host works with no chat workspace."""
    assert isinstance(report_sink("logs", None), LogSink)


def test_slack_builds_the_write_only_sink(monkeypatch, mocker):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-x")
    build_sink = mocker.patch("investigator.entrypoints.slack.post.build_sink")

    sink = report_sink("slack", "C1")

    build_sink.assert_called_once_with("xoxb-x", "C1")
    assert sink is build_sink.return_value


def test_slack_without_a_token_says_so(monkeypatch):
    """Checked before a webhook arrives, not when the first report is ready."""
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)

    with pytest.raises(ConfigError, match="SLACK_BOT_TOKEN"):
        report_sink("slack", "C1")


# --- the command line ---------------------------------------------------------


def test_slack_reporting_requires_a_channel(monkeypatch, capsys):
    _env(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main([*_ARGS, "--report", "slack"])

    assert exc.value.code == 2
    assert "--slack-channel" in capsys.readouterr().err


def test_a_missing_webhook_secret_names_itself(monkeypatch, mocker, capsys):
    _env(monkeypatch, BRONTO_WEBHOOK_SECRET=None)
    mocker.patch("investigator.entrypoints.webhook.main.build_model")
    serve = mocker.patch("investigator.entrypoints.webhook.main.serve")

    with pytest.raises(SystemExit) as exc:
        main(_ARGS)

    assert exc.value.code == 1
    assert "BRONTO_WEBHOOK_SECRET" in capsys.readouterr().err
    serve.assert_not_called()


@pytest.mark.parametrize("missing", ["BRONTO_API_KEY", "ANTHROPIC_API_KEY"])
def test_bronto_and_provider_config_are_still_validated(
    monkeypatch, mocker, capsys, missing
):
    _env(monkeypatch, **{missing: None})
    mocker.patch("investigator.entrypoints.webhook.main.build_model")
    serve = mocker.patch("investigator.entrypoints.webhook.main.serve")

    with pytest.raises(SystemExit) as exc:
        main(_ARGS)

    assert exc.value.code == 1
    assert missing in capsys.readouterr().err
    serve.assert_not_called()


def test_it_serves_with_everything_configured(monkeypatch, mocker):
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.webhook.main.build_model")
    # Autospec'd so a renamed parameter fails here rather than on a host.
    serve = mocker.patch("investigator.entrypoints.webhook.main.serve", autospec=True)

    main([*_ARGS, "--port", "9000", "--max-live", "1"])

    kwargs = serve.call_args.kwargs
    assert kwargs["secret"] == "shh"
    assert kwargs["port"] == 9000
    assert kwargs["max_live"] == 1
    # A factory, not one instance: concurrent runs must not share a sink.
    assert isinstance(kwargs["human_for"](), LogSink)
    assert kwargs["human_for"]() is not kwargs["human_for"]()


def test_no_vendor_of_any_kind_is_named_in_the_package():
    """The surface is host-agnostic: no cloud SDK, no metadata service, no
    control-plane call. Deploy config may name a host; `src/` may not."""
    package = Path(webhook.__file__).parent
    vendor = re.compile(
        r"\b(fly|machines\.dev|boto3|google\.cloud|azure|169\.254\.169\.254)\b"
    )

    named = [
        f"{path.name}: {match.group()}"
        for path in sorted(package.glob("*.py"))
        for match in vendor.finditer(path.read_text().lower())
    ]
    assert not named, named
