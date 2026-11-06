# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""LangGraph agent lifecycle telemetry example.

Runs a small human-in-the-loop LangGraph application twice, once until it
interrupts and once to resume it, under the LangChain instrumentor, and writes
every span and log record it produced to ``sample-output.json`` beside this
file. Needs no API key and no collector.

Usage, from this directory::

    python main.py

See `README.rst <README.rst>`_ for what the events mean and where they come
from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

OUTPUT = Path(__file__).with_name("sample-output.json")


class ExpenseState(TypedDict, total=False):
    amount: int
    approval: str
    status: str


def build_graph():
    def prepare(_state: ExpenseState) -> ExpenseState:
        return {"status": "prepared"}

    def await_approval(state: ExpenseState) -> ExpenseState:
        decision = interrupt(
            {"question": "approve expense?", "amount": state.get("amount")}
        )
        return {"approval": str(decision)}

    def submit(_state: ExpenseState) -> ExpenseState:
        return {"status": "submitted"}

    builder = StateGraph(ExpenseState)
    builder.add_node("prepare", prepare)
    builder.add_node("await_approval", await_approval)
    builder.add_node("submit", submit)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "await_approval")
    builder.add_edge("await_approval", "submit")
    builder.add_edge("submit", END)
    return builder.compile(checkpointer=InMemorySaver())


def _span_json(span: Any) -> dict[str, Any]:
    return {
        "name": span.name,
        "trace_id": f"{span.context.trace_id:032x}",
        "span_id": f"{span.context.span_id:016x}",
        "parent_span_id": (
            f"{span.parent.span_id:016x}" if span.parent else None
        ),
        "kind": str(span.kind),
        "attributes": dict(span.attributes or {}),
    }


def _log_json(record: Any) -> dict[str, Any]:
    return {
        "event_name": record.event_name,
        "body": record.body,
        "trace_id": f"{record.trace_id:032x}" if record.trace_id else None,
        "span_id": f"{record.span_id:016x}" if record.span_id else None,
        "attributes": dict(record.attributes or {}),
    }


def main() -> None:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(
        SimpleLogRecordProcessor(log_exporter)
    )

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider, logger_provider=logger_provider
    )
    try:
        graph = build_graph()
        config = {"configurable": {"thread_id": "expense-4711"}}
        paused = graph.invoke({"amount": 250}, config=config)
        interrupts = [
            {"id": item.id, "value": item.value}
            for item in paused["__interrupt__"]
        ]
        resumed = graph.invoke(Command(resume="approved"), config=config)
    finally:
        instrumentor.uninstrument()

    document = {
        "description": (
            "Telemetry captured by opentelemetry-instrumentation-genai-"
            "langchain from an unmodified LangGraph human-in-the-loop "
            "application: one invoke that interrupts, one that resumes."
        ),
        "langgraph_ground_truth": {
            "interrupts_returned_by_invoke": interrupts,
            "final_state": resumed,
        },
        "spans": [
            _span_json(span) for span in span_exporter.get_finished_spans()
        ],
        "logs": [
            _log_json(item.log_record)
            for item in log_exporter.get_finished_logs()
        ],
    }
    OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
