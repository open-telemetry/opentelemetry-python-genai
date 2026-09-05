# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LangGraph invocation cancellation handling.

Pins that cancelling an in-flight LangGraph invocation driven through the
public API records the workflow span as failed (StatusCode.ERROR) with
error.type="asyncio.exceptions.CancelledError", while properly propagating
the cancellation to the caller.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace.status import StatusCode

_TIMEOUT = 5.0


class _State(TypedDict):
    message: str


def test_langgraph_ainvoke_cancellation_records_error(
    span_exporter, start_instrumentation
) -> None:
    """A real awaited LangGraph ainvoke task cancelled by asyncio."""

    async def scenario() -> None:
        started = asyncio.Event()

        async def slow_step(state: _State) -> _State:
            started.set()
            await asyncio.sleep(60)
            return {"message": state["message"] + " done"}

        builder = StateGraph(_State)
        builder.add_node("slow_step", slow_step)
        builder.add_edge(START, "slow_step")
        builder.add_edge("slow_step", END)
        graph = builder.compile()

        task = asyncio.create_task(graph.ainvoke({"message": "test"}))
        await asyncio.wait_for(started.wait(), _TIMEOUT)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _TIMEOUT)

        assert task.cancelled()

    asyncio.run(scenario())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow LangGraph"
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == gen_ai_attributes.GenAiOperationNameValues.INVOKE_WORKFLOW.value
    )
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_WORKFLOW_NAME] == "LangGraph"
    )
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


def test_langgraph_astream_cancellation_records_error(
    span_exporter, start_instrumentation
) -> None:
    """A real awaited LangGraph astream task cancelled during iteration."""

    async def scenario() -> None:
        started = asyncio.Event()

        async def slow_step(state: _State) -> _State:
            started.set()
            await asyncio.sleep(60)
            return {"message": state["message"] + " done"}

        builder = StateGraph(_State)
        builder.add_node("slow_step", slow_step)
        builder.add_edge(START, "slow_step")
        builder.add_edge("slow_step", END)
        graph = builder.compile()

        async def consume() -> None:
            async for _ in graph.astream({"message": "test"}):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(started.wait(), _TIMEOUT)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _TIMEOUT)

        assert task.cancelled()

    asyncio.run(scenario())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow LangGraph"
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == gen_ai_attributes.GenAiOperationNameValues.INVOKE_WORKFLOW.value
    )
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_WORKFLOW_NAME] == "LangGraph"
    )
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


def test_langgraph_named_workflow_cancellation_records_error(
    span_exporter, start_instrumentation
) -> None:
    """A named LangGraph workflow preserves its name on cancellation."""

    async def scenario() -> None:
        started = asyncio.Event()

        async def slow_step(state: _State) -> _State:
            started.set()
            await asyncio.sleep(60)
            return {"message": state["message"] + " done"}

        builder = StateGraph(_State)
        builder.add_node("slow_step", slow_step)
        builder.add_edge(START, "slow_step")
        builder.add_edge("slow_step", END)
        graph = builder.compile()

        task = asyncio.create_task(
            graph.ainvoke(
                {"message": "test"},
                config={"metadata": {"workflow_name": "CustomGraph"}},
            )
        )
        await asyncio.wait_for(started.wait(), _TIMEOUT)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _TIMEOUT)

        assert task.cancelled()

    asyncio.run(scenario())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow CustomGraph"
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == gen_ai_attributes.GenAiOperationNameValues.INVOKE_WORKFLOW.value
    )
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_WORKFLOW_NAME]
        == "CustomGraph"
    )
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


def test_langgraph_multinode_cancellation_records_error(
    span_exporter, start_instrumentation
) -> None:
    """Cancellation in a downstream node of a multi-node workflow records error."""

    async def scenario() -> None:
        started = asyncio.Event()

        async def step_one(state: _State) -> _State:
            return {"message": state["message"] + " step1"}

        async def step_two(state: _State) -> _State:
            started.set()
            await asyncio.sleep(60)
            return {"message": state["message"] + " step2"}

        builder = StateGraph(_State)
        builder.add_node("step_one", step_one)
        builder.add_node("step_two", step_two)
        builder.add_edge(START, "step_one")
        builder.add_edge("step_one", "step_two")
        builder.add_edge("step_two", END)
        graph = builder.compile()

        task = asyncio.create_task(graph.ainvoke({"message": "test"}))
        await asyncio.wait_for(started.wait(), _TIMEOUT)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, _TIMEOUT)

        assert task.cancelled()

    asyncio.run(scenario())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow LangGraph"
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == gen_ai_attributes.GenAiOperationNameValues.INVOKE_WORKFLOW.value
    )
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_WORKFLOW_NAME] == "LangGraph"
    )
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


def test_langgraph_clean_invocation_leaves_status_unset(
    span_exporter, start_instrumentation
) -> None:
    """Regression baseline: clean execution leaves status UNSET without error.type."""

    async def scenario() -> None:
        async def fast_step(state: _State) -> _State:
            return {"message": state["message"] + " done"}

        builder = StateGraph(_State)
        builder.add_node("fast_step", fast_step)
        builder.add_edge(START, "fast_step")
        builder.add_edge("fast_step", END)
        graph = builder.compile()

        result = await graph.ainvoke({"message": "test"})
        assert result["message"] == "test done"

    asyncio.run(scenario())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_workflow LangGraph"
    assert span.status.status_code == StatusCode.UNSET
    assert (
        span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == gen_ai_attributes.GenAiOperationNameValues.INVOKE_WORKFLOW.value
    )
    assert error_attributes.ERROR_TYPE not in span.attributes
