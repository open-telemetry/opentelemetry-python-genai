# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from opentelemetry import _logs, metrics, trace
from opentelemetry._logs import LogRecord
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)


class PrintCompletionHook(CompletionHook):
    """Print captured inputs and outputs instead of forwarding them externally."""

    def on_completion(
        self,
        *,
        inputs: list[InputMessage],
        outputs: list[OutputMessage],
        system_instruction: list[MessagePart],
        tool_definitions: list[ToolDefinition] | None = None,
        span: Span | None = None,
        log_record: LogRecord | None = None,
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
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
    )
)

LlamaIndexInstrumentor().instrument(completion_hook=PrintCompletionHook())
OpenAIInstrumentor().instrument(completion_hook=PrintCompletionHook())


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It is sunny in {city}."


async def main() -> None:
    agent = FunctionAgent(
        name="assistant",
        llm=OpenAI(model=os.getenv("CHAT_MODEL", "gpt-4o-mini")),
        tools=[FunctionTool.from_defaults(get_weather)],
        streaming=False,
    )
    response = await agent.run(
        user_msg="Use get_weather to check the weather in Paris."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
