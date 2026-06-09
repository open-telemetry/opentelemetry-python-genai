# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Single-node agent example built with StateGraph.

A single ReAct agent answers arithmetic questions using calculator tools.
OpenTelemetry LangChain instrumentation traces all LLM calls.
"""

from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
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
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# Configure logging
_logs.set_logger_provider(LoggerProvider())
_logs.get_logger_provider().add_log_record_processor(
    BatchLogRecordProcessor(OTLPLogExporter())
)

# Configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())]
    )
)


# --- Tools ----------------------------------------------------------------


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


# --- Graph ----------------------------------------------------------------


def build_single_node_graph(llm: ChatOpenAI):
    session_id = str(uuid4())

    agent = create_react_agent(
        llm, tools=[multiply, add], name="math_agent"
    ).with_config(
        {
            "metadata": {
                "agent_name": "math_agent",
                "session_id": session_id,
            },
        }
    )

    def run_agent(state: MessagesState) -> dict:
        result = agent.invoke({"messages": state["messages"]})
        return {"messages": result["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("math_agent", run_agent)
    builder.add_edge(START, "math_agent")
    builder.add_edge("math_agent", END)

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, seed=100)
    graph = build_single_node_graph(llm)

    questions = [
        "What is 12 multiplied by 7?",
        "What is 15 plus 27?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = graph.invoke({"messages": [HumanMessage(content=question)]})
        last = result["messages"][-1]
        print(f"  Answer: {last.content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
