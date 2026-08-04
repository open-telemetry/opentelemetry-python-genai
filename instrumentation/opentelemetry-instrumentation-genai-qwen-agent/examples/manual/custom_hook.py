# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
"""Instruments Qwen-Agent with a custom CompletionHook that prints prompts
and completions to stdout.

Requires DASHSCOPE_API_KEY to be set.

Run with: python custom_hook.py
"""

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
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)


class PrintCompletionHook(CompletionHook):
    """Minimal CompletionHook that prints inputs/outputs to stdout.

    Real hooks typically forward content to external storage (object store,
    database, etc.) and record reference URIs on the span/log_record.
    """

    def on_completion(
        self,
        *,
        inputs: list[InputMessage],
        outputs: list[OutputMessage],
        system_instruction: list[MessagePart],
        tool_definitions: list[ToolDefinition] | None = None,
        span=None,
        log_record=None,
    ) -> None:
        print(f"[hook] inputs: {inputs}")
        print(f"[hook] outputs: {outputs}")


trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter()),
        ]
    )
)

QwenAgentInstrumentor().instrument(completion_hook=PrintCompletionHook())


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
