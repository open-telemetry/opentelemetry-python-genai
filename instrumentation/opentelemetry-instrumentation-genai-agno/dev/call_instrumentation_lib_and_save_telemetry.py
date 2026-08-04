# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from agno.agent import Agent
from agno.models.google import Gemini
from google.protobuf import text_format

from opentelemetry import _logs as logs
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common._internal import (
    _log_encoder,
    trace_encoder,
)
from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

resource = Resource.create(
    attributes={SERVICE_NAME: "agno-write-to-file-example"}
)
in_memory_log_exporter = InMemoryLogRecordExporter()
in_memory_span_exporter = InMemorySpanExporter()

trace_provider = TracerProvider(resource=resource)
trace_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
trace.set_tracer_provider(trace_provider)
logger_provider = LoggerProvider(resource=resource)
logger_provider.add_log_record_processor(
    SimpleLogRecordProcessor(in_memory_log_exporter)
)
logs.set_logger_provider(logger_provider)


def write_logs_to_file(file_name: str):
    log_records = in_memory_log_exporter.get_finished_logs()
    log_proto = _log_encoder.encode_logs(log_records)
    with open(
        os.path.join(os.getcwd(), file_name + ".textproto"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text_format.MessageToString(log_proto))


def write_spans_to_file(file_name: str):
    spans = in_memory_span_exporter.get_finished_spans()
    span_proto = trace_encoder.encode_spans(spans)
    with open(
        os.path.join(os.getcwd(), file_name + ".textproto"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(text_format.MessageToString(span_proto))


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def main():
    GoogleGenAiSdkInstrumentor().instrument()
    AgnoInstrumentor().instrument()

    agent = Agent(
        model=Gemini(
            id=os.environ.get("MODEL", "gemini-2.5-flash"),
        ),
        tools=[add],
        name="google-genai-agent",
    )

    response = agent.run(
        os.environ.get("PROMPT", "What is 123 + 456? Use the tool to compute.")
    )

    write_spans_to_file("span_agno_google_genai")
    write_logs_to_file("log_agno_google_genai")
    print(response)


if __name__ == "__main__":
    main()
