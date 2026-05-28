# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
ReAct agent example built with LangGraph.

Uses LangGraph's prebuilt create_react_agent with simple calculator tools.
OpenTelemetry LangChain instrumentation traces the LLM calls made by the agent.
"""

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        top_p=0.9,
        seed=100,
    )

    session_id = str(uuid4())
    agent = create_react_agent(llm, tools=[multiply, add]).with_config(
        {
            "metadata": {
                "agent_name": "coordinator",
                "session_id": session_id,
            },
        }
    )

    result = agent.invoke(
        {"messages": [HumanMessage(content="What is (3 * 4) + 7?")]}
    )

    print("Agent output:")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
