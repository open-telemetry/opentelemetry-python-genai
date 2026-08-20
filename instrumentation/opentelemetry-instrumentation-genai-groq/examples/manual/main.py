# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import os

from groq import Groq

# NOTE: OpenTelemetry Python Logs API is in beta
from opentelemetry import _logs, metrics, trace
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.groq import GroqInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# configure logging
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
            ),
        ]
    )
)

# instrument Groq
GroqInstrumentor().instrument()


def main():
    client = Groq()
    chat_completion = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "llama3-8b-8192"),
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
