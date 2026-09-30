"""Tests for the CLI adapter: argument parsing, env validation, and wiring.

Rendering prompts, parsing typed answers, mapping them to decisions, and the
interrupt loop live behind the Human seam and are tested in test_human.py,
test_terminal.py and test_run.py.
"""

import io
import json

import pytest
from bronto_sdk import BrontoClient

from investigator.entrypoints.cli.main import investigate, main
from investigator.entrypoints.cli.terminal import TerminalHuman
from investigator.settings import DEFAULT_MAX_MODEL_CALLS

_ARGS = ["--provider", "anthropic", "--model-id", "claude-x"]


def _env(monkeypatch, **overrides):
    values: dict[str, str | None] = {
        "BRONTO_API_KEY": "secret-key",
        "BRONTO_REGION": "eu",
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
    }
    values.update(overrides)
    for key, value in values.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


# --- arguments ----------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,flag",
    [
        (["--model-id", "claude-x"], "--provider"),
        (["--provider", "anthropic"], "--model-id"),
        (["--provider", "nonesuch", "--model-id", "claude-x"], "invalid choice"),
    ],
    ids=["no provider", "no model id", "unknown provider"],
)
def test_cli_rejects_bad_arguments_by_name(monkeypatch, capsys, argv, flag):
    """Asserted on the message, not just a non-zero exit.

    An exit-code-only assertion passes for *any* argparse rejection, including
    a flag that no longer exists — which is how two earlier tests went on
    passing after their flag was deleted.
    """
    _env(monkeypatch)

    with pytest.raises(SystemExit) as exc:
        main(argv)

    assert exc.value.code == 2
    assert flag in capsys.readouterr().err


def test_cli_threads_max_model_calls_to_run(monkeypatch, mocker):
    """--max-model-calls is parsed and passed through to run."""
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    run = mocker.patch("investigator.entrypoints.cli.main.run")

    main([*_ARGS, "--max-model-calls", "3"])

    assert run.call_args.kwargs["settings"].max_model_calls == 3


def test_cli_max_model_calls_defaults_to_the_shared_budget(monkeypatch, mocker):
    """Without the flag, the model-call budget falls back to the graph default."""
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    run = mocker.patch("investigator.entrypoints.cli.main.run")

    main(_ARGS)

    assert run.call_args.kwargs["settings"].max_model_calls == DEFAULT_MAX_MODEL_CALLS


# --- configuration ------------------------------------------------------------


@pytest.mark.parametrize("missing", ["BRONTO_API_KEY", "BRONTO_REGION"])
def test_cli_names_the_missing_bronto_variable(monkeypatch, capsys, missing):
    """Even with a valid provider and its env, missing Bronto env still exits."""
    _env(monkeypatch, **{missing: None})

    with pytest.raises(SystemExit) as exc:
        main(_ARGS)

    assert exc.value.code == 1
    assert missing in capsys.readouterr().err


def test_cli_exits_when_provider_env_missing(monkeypatch, mocker, capsys):
    """A valid provider with its env var missing exits before building anything."""
    _env(monkeypatch, OPENAI_API_KEY=None)
    build_model = mocker.patch("investigator.entrypoints.cli.main.build_model")

    with pytest.raises(SystemExit) as exc:
        main(["--provider", "openai", "--model-id", "gpt-5"])

    assert exc.value.code == 1
    build_model.assert_not_called()
    err = capsys.readouterr().err
    assert "OPENAI_API_KEY" in err
    assert "openai" in err


def test_cli_builds_model_for_chosen_provider(monkeypatch, mocker):
    """main builds the model for the chosen provider + model id via build_model."""
    _env(monkeypatch, AWS_REGION="eu-west-1", AWS_PROFILE="sandbox")
    build_model = mocker.patch("investigator.entrypoints.cli.main.build_model")
    mocker.patch("investigator.entrypoints.cli.main.run")

    main(["--provider", "bedrock", "--model-id", "anthropic.claude-sonnet-4"])

    build_model.assert_called_once_with("bedrock", "anthropic.claude-sonnet-4")


def test_cli_passes_built_model_to_run(monkeypatch, mocker):
    """Happy path: main builds the model and hands it (with the client) to run."""
    _env(monkeypatch)
    model = object()
    mocker.patch("investigator.entrypoints.cli.main.build_model", return_value=model)
    run = mocker.patch("investigator.entrypoints.cli.main.run")

    main(_ARGS)

    run.assert_called_once()
    client, passed_model = run.call_args.args[0], run.call_args.args[1]
    assert isinstance(client, BrontoClient)
    assert passed_model is model


@pytest.mark.parametrize("endpoint", ["https://otel.example.io", None])
def test_cli_inits_telemetry_only_when_an_otel_endpoint_is_set(
    monkeypatch, mocker, endpoint
):
    _env(monkeypatch, OTEL_EXPORTER_OTLP_ENDPOINT=endpoint)
    init_telemetry = mocker.patch("investigator.entrypoints.cli.main.init_telemetry")
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    mocker.patch("investigator.entrypoints.cli.main.run")

    main(_ARGS)

    assert init_telemetry.call_count == (1 if endpoint else 0)


# --- the interactive path -----------------------------------------------------


async def test_cli_places_alert_input_into_initial_state(mocker):
    """The natural-language alert typed by the user seeds the graph's state."""
    app = mocker.Mock()
    app.ainvoke = mocker.AsyncMock()
    app.ainvoke.return_value = {}  # completes without an interrupt

    human = TerminalHuman(
        input_fn=lambda: "checkout error rate is high", output_fn=lambda _: None
    )

    await investigate(app, human, None)

    state = app.ainvoke.call_args.args[0]
    assert state["request_text"] == "checkout error rate is high"


# --- the event path -----------------------------------------------------------


_EVENT = {
    "monitor_id": "mon-1",
    "monitor_name": "foo bar baz",
    "triggered_at": "2026-07-24T14:19:09Z",
    "evaluation_timestamp": 1784901720000,
}


def test_cli_loads_an_event_from_a_file(monkeypatch, mocker, tmp_path):
    """--event feeds an already-identified alert straight in, so the run never
    prompts and never searches for one."""
    _env(monkeypatch)
    path = tmp_path / "event.json"
    path.write_text(json.dumps(_EVENT))
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    run = mocker.patch("investigator.entrypoints.cli.main.run")

    main([*_ARGS, "--event", str(path)])

    assert run.call_args.kwargs["event"] == _EVENT


def test_cli_reads_an_event_from_stdin(monkeypatch, mocker):
    """'-' is the piping case: `curl ... | investigator --event -`."""
    _env(monkeypatch)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_EVENT)))
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    run = mocker.patch("investigator.entrypoints.cli.main.run")

    main([*_ARGS, "--event", "-"])

    assert run.call_args.kwargs["event"] == _EVENT


def test_cli_reports_an_unreadable_event_rather_than_traceback(
    monkeypatch, mocker, capsys
):
    _env(monkeypatch)
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    mocker.patch("investigator.entrypoints.cli.main.run")

    with pytest.raises(SystemExit) as exc:
        main([*_ARGS, "--event", "/no/such/file.json"])

    assert exc.value.code == 1
    assert "could not read --event" in capsys.readouterr().err


def test_cli_rejects_an_event_that_is_not_a_json_object(
    monkeypatch, mocker, tmp_path, capsys
):
    """A JSON array is valid JSON but not an event payload."""
    _env(monkeypatch)
    path = tmp_path / "event.json"
    path.write_text("[1, 2, 3]")
    mocker.patch("investigator.entrypoints.cli.main.build_model")
    mocker.patch("investigator.entrypoints.cli.main.run")

    with pytest.raises(SystemExit) as exc:
        main([*_ARGS, "--event", str(path)])

    assert exc.value.code == 1
    assert "could not read --event" in capsys.readouterr().err


async def test_investigating_an_event_never_prompts(mocker):
    """The whole point of the event path: no description asked for, no
    interrupt serviced, and the graph entered with the triggered alert already set."""
    app = mocker.Mock()
    app.ainvoke = mocker.AsyncMock(return_value={"report": "done"})
    outputs: list[str] = []

    def never_called():
        raise AssertionError("the event path must not read from stdin")

    human = TerminalHuman(input_fn=never_called, output_fn=outputs.append)

    await investigate(app, human, _EVENT)

    state = app.ainvoke.call_args.args[0]
    assert state["triggered_alert"]["id"] == "mon-1"
    assert state["request_text"] is None
    assert outputs == ["done"]


def test_an_unusable_event_payload_exits_non_zero(
    monkeypatch, mocker, tmp_path, capsys
):
    """What a queue worker branches on. Validated at the edge, so no MCP
    session is opened and no model call is paid for."""
    _env(monkeypatch)
    build = mocker.patch("investigator.entrypoints.cli.main.build_model")
    path = tmp_path / "event.json"
    path.write_text(json.dumps({"monitor_name": "no id, no timestamp"}))

    with pytest.raises(SystemExit) as exc:
        main([*_ARGS, "--event", str(path)])

    assert exc.value.code == 1
    assert "unusable --event payload" in capsys.readouterr().err
    build.assert_not_called()
