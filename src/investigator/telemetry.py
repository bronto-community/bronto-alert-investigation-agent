"""OpenTelemetry tracing for the agent (OpenInference + OTLP export).

Auto-instruments LangChain/LangGraph so every model call and tool call (with
inputs and outputs) becomes an OTel span, exported via OTLP to whatever
`OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` point at.
"""

import logging

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "investigator"


class _DropMissingTracerCallbackWarning(logging.Filter):
    """Silence LangChain warnings for callbacks the OpenInference tracer lacks.

    Newer LangGraph fires callbacks (e.g. on_interrupt, on_resume) the installed
    tracer version doesn't implement; LangChain logs a harmless warning each time.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("OpenInferenceTracer" in message and "has no attribute" in message)


def init_telemetry() -> None:
    """Configure OTLP tracing and instrument LangChain. Idempotent per process."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()  # Bronto REST and MCP calls become spans
    logging.getLogger("langchain_core.callbacks.manager").addFilter(
        _DropMissingTracerCallbackWarning()
    )
