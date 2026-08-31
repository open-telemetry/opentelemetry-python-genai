# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

import pytest

InMemorySaver = pytest.importorskip(
    "langgraph.checkpoint.memory"
).InMemorySaver
_graph = pytest.importorskip("langgraph.graph")
END = _graph.END
START = _graph.START
StateGraph = _graph.StateGraph


class _State(TypedDict, total=False):
    approval_status: str


def test_persisted_node_delta_emits_correlated_state_event(
    start_instrumentation,
    log_exporter,
    span_exporter,
):
    def approve(state: _State) -> _State:
        return {"approval_status": "accepted"}

    builder = StateGraph(_State)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "state-event-test"}}

    assert graph.invoke({}, config=config) == {"approval_status": "accepted"}
    assert graph.get_state(config).values == {"approval_status": "accepted"}

    state_events = [
        item.log_record
        for item in log_exporter.get_finished_logs()
        if item.log_record.event_name == "gen_ai.execution.state.changed"
    ]
    assert len(state_events) == 1
    event = state_events[0]
    assert event.attributes is not None
    assert event.attributes["gen_ai.execution.state.changed_key.count"] == 1
    assert "gen_ai.execution.state.version" in event.attributes
    assert "gen_ai.execution.state.changed_keys" not in event.attributes
    assert "approval_status" not in event.attributes.values()
    assert "accepted" not in event.attributes.values()

    workflow_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get("gen_ai.operation.name") == "invoke_workflow"
    ]
    assert len(workflow_spans) == 1
    assert event.trace_id == workflow_spans[0].context.trace_id
    assert event.span_id == workflow_spans[0].context.span_id
