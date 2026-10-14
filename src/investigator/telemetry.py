"""OpenTelemetry tracing for the agent (GenAI LangChain instrumentation + OTLP).

Auto-instruments LangChain/LangGraph so every graph, agent, model call and tool
call becomes an OTel span, exported via OTLP to whatever
`OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_HEADERS` point at.
"""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

SERVICE_NAME = "investigator"

# The instrumentation leaves prompts and completions out unless asked; the
# decision chain is what these traces are for, so ask, on the span itself.
_CAPTURE_CONTENT_ENV = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"


def init_telemetry() -> None:
    """Configure OTLP tracing and instrument LangChain. Idempotent per process."""
    os.environ.setdefault(_CAPTURE_CONTENT_ENV, "SPAN_ONLY")
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()  # Bronto REST and MCP calls become spans
