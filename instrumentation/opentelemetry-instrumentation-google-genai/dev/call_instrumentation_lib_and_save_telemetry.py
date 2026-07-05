# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from google import genai
from google.genai import types
from google.protobuf import text_format

from opentelemetry import _logs as logs
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.common._internal import (
    _log_encoder,
    trace_encoder,
)
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

resource = Resource.create(attributes={SERVICE_NAME: "write-to-file-example"})
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
    logs = in_memory_log_exporter.get_finished_logs()
    log_proto = _log_encoder.encode_logs(logs)
    with open(os.path.join(os.getcwd(), file_name + ".textproto"), "w") as f:
        f.write(text_format.MessageToString(log_proto))


def write_spans_to_file(file_name: str):
    spans = in_memory_span_exporter.get_finished_spans()
    span_proto = trace_encoder.encode_spans(spans)
    with open(os.path.join(os.getcwd(), file_name + ".textproto"), "w") as f:
        f.write(text_format.MessageToString(span_proto))


def add(a: int, b: int) -> int:
    return a + b


def main():
    GoogleGenAiSdkInstrumentor().instrument()
    # False for the embedding call
    client = genai.Client(
        vertexai=True,
        project=os.environ["PROJECT_ID"],
        location=os.environ["LOCATION"],
    )
    response = client.models.generate_content(
        model=os.environ["MODEL"],
        contents=os.environ["PROMPT"],
        config=types.GenerateContentConfig(tools=[add]),
    )
    # embed_response = client.models.embed_content(
    #     model=os.environ["MODEL"],
    #     contents=os.environ["PROMPT"],
    # )
    write_spans_to_file("test_span")
    write_logs_to_file("test_log")
    print(response.text)


if __name__ == "__main__":
    main()
