# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Tool-calling example without agents, built with LangChain.

Uses ChatOpenAI with bind_tools to let the model call calculator tools directly,
then manually dispatches tool calls and feeds results back to the model.
OpenTelemetry LangChain instrumentation traces the LLM calls.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

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
from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
span_processor = BatchSpanProcessor(OTLPSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Configure logging
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# Configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(
                OTLPMetricExporter(),
            ),
        ]
    )
)


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


TOOLS = [multiply, add]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def main() -> None:
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        seed=100,
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [HumanMessage(content="What is (3 * 4) + 7?")]

    # First LLM call — model may request tool calls
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # Dispatch tool calls until the model stops requesting them
    while response.tool_calls:
        for tool_call in response.tool_calls:
            selected_tool = TOOLS_BY_NAME[tool_call["name"]]
            tool_output = selected_tool.invoke(tool_call["args"])
            messages.append(
                ToolMessage(
                    content=json.dumps(tool_output),
                    tool_call_id=tool_call["id"],
                )
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    print("Final answer:", response.content)

    instrumentor.uninstrument()


if __name__ == "__main__":
    main()
