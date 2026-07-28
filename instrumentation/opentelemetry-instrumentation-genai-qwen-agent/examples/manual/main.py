# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import os

from qwen_agent.agents import Assistant

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
from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
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

# configure logging and events
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter()),
        ]
    )
)

# instrument Qwen-Agent
QwenAgentInstrumentor().instrument()


def main():
    bot = Assistant(
        llm={
            "model": os.getenv("CHAT_MODEL", "qwen-max"),
            "model_type": "qwen_dashscope",
        },
        name="example-assistant",
        system_message="You are a helpful assistant.",
    )
    responses = []
    for responses in bot.run(
        [{"role": "user", "content": "Write a short poem on OpenTelemetry."}]
    ):
        pass
    for message in responses:
        print(message)


if __name__ == "__main__":
    main()
