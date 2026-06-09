# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LangGraph StateGraph example with two LLM nodes.

Graph topology:

  START → researcher → summariser → END

Steps:
  1. *researcher*  – gathers factual background on the user's question.
  2. *summariser*  – condenses the researcher's output into a concise answer.

OpenTelemetry LangChain instrumentation traces both LLM calls.
"""

from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

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


class GraphState(TypedDict):
    """State shared across all graph nodes."""

    messages: Annotated[list, add_messages]
    research: str


def build_graph(llm: ChatOpenAI):
    """Build a StateGraph with a researcher node and a summariser node."""

    def researcher(state: GraphState) -> dict:
        """Gather factual background on the last user message."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are a research assistant. Provide 2-3 factual sentences."
                ),
                HumanMessage(content=state["messages"][-1].content),
            ]
        )
        return {
            "research": response.content,
            "messages": [response],
        }

    def summariser(state: GraphState) -> dict:
        """Condense the researcher's output into one concise sentence."""
        response = llm.invoke(
            [
                SystemMessage(
                    content="You are an expert summariser. Condense the text below into one clear sentence."
                ),
                HumanMessage(content=state["research"]),
            ]
        )
        return {"messages": [response]}

    builder = StateGraph(GraphState)
    builder.add_node("researcher", researcher)
    builder.add_node("summariser", summariser)

    builder.add_edge(START, "researcher")
    builder.add_edge("researcher", "summariser")
    builder.add_edge("summariser", END)

    return builder.compile()


def main():
    LangChainInstrumentor().instrument()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=200,
        seed=42,
    )

    graph = build_graph(llm)

    question = "What is the capital of France?"
    print(f"Question: {question}\n")

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "research": "",
        }
    )

    print("Research output:")
    print(f"  {result['research']}\n")

    print("Final summary:")
    print(f"  {result['messages'][-1].content}")

    LangChainInstrumentor().uninstrument()


if __name__ == "__main__":
    main()
