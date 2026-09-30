"""Tests for OpenTelemetry / OpenInference wiring."""

import logging

from investigator.telemetry import (
    _DropMissingTracerCallbackWarning,
    init_telemetry,
)


def _record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        "langchain_core.callbacks.manager", logging.WARNING, __file__, 0, msg, (), None
    )


def test_filter_drops_missing_tracer_callback_warnings():
    """Drops OpenInferenceTracer missing-callback warnings, keeps unrelated ones."""
    log_filter = _DropMissingTracerCallbackWarning()

    on_interrupt = _record(
        'Error in OpenInferenceTracer.on_interrupt callback: AttributeError("'
        "'OpenInferenceTracer' object has no attribute 'on_interrupt'\")"
    )
    on_resume = _record(
        'Error in OpenInferenceTracer.on_resume callback: AttributeError("'
        "'OpenInferenceTracer' object has no attribute 'on_resume'\")"
    )
    unrelated = _record("Error in SomeHandler.on_llm_end callback: ValueError('boom')")

    assert log_filter.filter(on_interrupt) is False
    assert log_filter.filter(on_resume) is False
    assert log_filter.filter(unrelated) is True


def test_init_telemetry_instruments_langchain_and_silences_its_warnings(
    mocker, monkeypatch
):
    """A tracer provider exporting over OTLP, LangChain and HTTPX instrumented,
    and exactly one warning filter on the LangChain callbacks logger."""
    callbacks = logging.getLogger("langchain_core.callbacks.manager")
    monkeypatch.setattr(callbacks, "filters", [])
    mocker.patch("investigator.telemetry.TracerProvider")
    mocker.patch("investigator.telemetry.BatchSpanProcessor")
    mocker.patch("investigator.telemetry.OTLPSpanExporter")
    set_provider = mocker.patch("investigator.telemetry.trace.set_tracer_provider")
    langchain = mocker.patch("investigator.telemetry.LangChainInstrumentor")
    httpx = mocker.patch("investigator.telemetry.HTTPXClientInstrumentor")

    init_telemetry()

    set_provider.assert_called_once()
    langchain.return_value.instrument.assert_called_once()
    httpx.return_value.instrument.assert_called_once()
    assert [type(f) for f in callbacks.filters] == [_DropMissingTracerCallbackWarning]
