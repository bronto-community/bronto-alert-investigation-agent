"""Tests for OpenTelemetry / GenAI LangChain instrumentation wiring."""

import os

import pytest

from investigator.telemetry import init_telemetry

_CAPTURE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


@pytest.fixture
def otel(mocker):
    """Patch out the real provider, exporter and instrumentors."""
    mocker.patch("investigator.telemetry.TracerProvider")
    mocker.patch("investigator.telemetry.BatchSpanProcessor")
    mocker.patch("investigator.telemetry.OTLPSpanExporter")
    return {
        "set_provider": mocker.patch(
            "investigator.telemetry.trace.set_tracer_provider"
        ),
        "langchain": mocker.patch("investigator.telemetry.LangChainInstrumentor"),
        "httpx": mocker.patch("investigator.telemetry.HTTPXClientInstrumentor"),
    }


def test_init_telemetry_instruments_langchain_and_httpx(otel, monkeypatch):
    """A tracer provider exporting over OTLP, LangChain and HTTPX instrumented,
    and prompts and completions captured on the spans."""
    monkeypatch.delenv(_CAPTURE, raising=False)

    init_telemetry()

    otel["set_provider"].assert_called_once()
    otel["langchain"].return_value.instrument.assert_called_once()
    otel["httpx"].return_value.instrument.assert_called_once()
    assert os.environ[_CAPTURE] == "SPAN_ONLY"


def test_init_telemetry_keeps_an_operator_content_setting(otel, monkeypatch):
    monkeypatch.setenv(_CAPTURE, "NO_CONTENT")

    init_telemetry()

    assert os.environ[_CAPTURE] == "NO_CONTENT"
