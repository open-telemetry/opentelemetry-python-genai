# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict
from uuid import uuid4

import pytest
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.util.genai.handler import TelemetryHandler

InMemorySaver = pytest.importorskip(
    "langgraph.checkpoint.memory"
).InMemorySaver
_graph = pytest.importorskip("langgraph.graph")
END = _graph.END
START = _graph.START
StateGraph = _graph.StateGraph
Command = pytest.importorskip("langgraph.types").Command


class _State(TypedDict, total=False):
    approval_status: str


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["invoke", "ainvoke", "stream", "astream"])
@pytest.mark.parametrize("output_kind", ["mapping", "command", "nested"])
async def test_persisted_node_delta_emits_correlated_state_event(
    start_instrumentation,
    log_exporter,
    span_exporter,
    mode,
    output_kind,
):
    def approve(state: _State):
        update = {"approval_status": "accepted"}
        if output_kind == "command":
            return Command(update=update, goto=END)
        if output_kind == "nested":
            return (
                RunnableLambda(lambda _: update)
                .with_config(run_name="approve")
                .invoke(state)
            )
        return update

    builder = StateGraph(_State)
    builder.add_node("approve", approve)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "state-event-test"}}
    initial = {"approval_status": "pending"}

    if mode == "ainvoke":
        result = await graph.ainvoke(initial, config=config)
    elif mode == "invoke":
        result = graph.invoke(initial, config=config)
    elif mode == "stream":
        result = list(
            graph.stream(initial, config=config, stream_mode="updates")
        )
    else:
        result = [
            delta
            async for delta in graph.astream(
                initial, config=config, stream_mode="updates"
            )
        ]
    expected = {"approval_status": "accepted"}
    assert result == (
        [{"approve": expected}] if mode in ("stream", "astream") else expected
    )
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
    assert "gen_ai.execution.state.version" not in event.attributes
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


@pytest.mark.parametrize(
    "outputs",
    [
        None,
        "value",
        42,
        ["value"],
        Command(),
        Command(update=[("key", "value")]),
    ],
)
def test_non_mapping_node_outputs_do_not_prevent_cleanup(
    tracer_provider, logger_provider, meter_provider, log_exporter, outputs
):
    handler = OpenTelemetryLangChainCallbackHandler(
        TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        )
    )
    workflow_id, node_id = uuid4(), uuid4()
    handler.on_chain_start(
        {"name": "LangGraph"},
        {},
        run_id=workflow_id,
        metadata={"ls_integration": "langgraph"},
    )
    handler.on_chain_start(
        {"name": "node"},
        {},
        run_id=node_id,
        parent_run_id=workflow_id,
        metadata={"ls_integration": "langgraph", "langgraph_node": "node"},
        tags=["graph:step:1"],
    )
    handler.on_chain_end(outputs, run_id=node_id, parent_run_id=workflow_id)
    handler.on_chain_end({}, run_id=workflow_id)

    assert handler._invocation_manager._invocations == {}
    assert not any(
        record.log_record.event_name == "gen_ai.execution.state.changed"
        for record in log_exporter.get_finished_logs()
    )
