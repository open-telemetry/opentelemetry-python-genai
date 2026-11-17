# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent telemetry exercised through LangChain's public APIs, without network calls."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.prebuilt import ToolNode, create_react_agent
from tests._api import FakeChatModel, FakeRetriever, finished_span
from tests._api import invoke as _invoke

from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["invoke", "stream"])
@pytest.mark.parametrize(
    ("failure", "use_tool", "retry"),
    [
        pytest.param(None, False, False, id="model-only"),
        pytest.param(None, True, False, id="with-tool"),
        pytest.param("model", True, False, id="model-error"),
        pytest.param("tool", True, False, id="tool-error"),
        pytest.param(None, False, True, id="retry"),
    ],
)
async def test_native_agent_call_counts(
    asynchronous: bool,
    streaming: bool,
    failure: str | None,
    use_tool: bool,
    retry: bool,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_langchain: None,
) -> None:
    error = RuntimeError("agent call failed")

    @tool
    def calculate() -> str:
        """Return the calculation result."""
        if failure == "tool":
            raise error
        return "42"

    model = FakeChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "calculate", "args": {}, "id": "1"}],
            ),
            AIMessage(content="42"),
        ]
        if use_tool
        else [AIMessage(content="42")],
        error=error if failure == "model" or retry else None,
    )

    class RecoverOnError(BaseCallbackHandler):
        def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
            model.error = None

    config: RunnableConfig = {"callbacks": [RecoverOnError()]} if retry else {}
    runnable = (
        model.with_retry(wait_exponential_jitter=False, stop_after_attempt=2)
        if retry
        else model
    )
    agent = create_react_agent(
        runnable,
        ToolNode([calculate], handle_tool_errors=False) if use_tool else [],
        name="counter-agent",
    )
    inputs = {"messages": [("user", "calculate")]}

    async def run() -> None:
        if not streaming:
            result = await _invoke(agent, inputs, asynchronous, config=config)
            assert result["messages"][-1].content == "42"
        elif asynchronous:
            async for _ in agent.astream(inputs, config=config):
                pass
        else:
            list(agent.stream(inputs, config=config))

    if failure:
        with pytest.raises(RuntimeError) as raised:
            await run()
        assert raised.value is error
    else:
        await run()
    spans = span_exporter.get_finished_spans()
    agent_span = finished_span(span_exporter, "invoke_agent counter-agent")
    assert agent_span.parent is None
    assert agent_span.status.status_code == (
        StatusCode.ERROR if failure else StatusCode.UNSET
    )
    if failure:
        assert (
            agent_span.attributes[error_attributes.ERROR_TYPE]
            == "RuntimeError"
        )
    inference_calls = 2 if retry or (use_tool and not failure) else 1
    chats = [
        span
        for span in spans
        if span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "chat"
    ]
    assert len(chats) == inference_calls
    if retry:
        assert [span.status.status_code for span in chats] == [
            StatusCode.ERROR,
            StatusCode.UNSET,
        ]
        assert (
            chats[0].attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
        )
    _assert_agent_counts(
        metric_reader,
        {
            "counter-agent": (
                inference_calls,
                int(use_tool and failure != "model"),
            )
        },
    )


def _assert_agent_counts(
    metric_reader: InMemoryMetricReader,
    expected: dict[str, tuple[int, int]],
) -> None:
    data = metric_reader.get_metrics_data()
    assert data is not None
    metrics = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    for index, name in enumerate(
        (
            "gen_ai.invoke_agent.inference_calls",
            "gen_ai.invoke_agent.tool_calls",
        )
    ):
        points = metrics[name].data.data_points
        assert len(points) == len(expected)
        assert {
            point.attributes[gen_ai_attributes.GEN_AI_AGENT_NAME]: point.sum
            for point in points
        } == {agent: counts[index] for agent, counts in expected.items()}
        for point in points:
            assert point.count == 1
            assert type(point.sum) is int


def _agent_with_tool(agent_tool: BaseTool, name: str) -> Runnable:
    return create_react_agent(
        FakeChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": agent_tool.name, "args": {}, "id": "1"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        ),
        [agent_tool],
        name=name,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_native_agent_counter_ownership(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_langchain: None,
) -> None:
    inputs = {"messages": [("user", "hi")]}
    inner = create_react_agent(
        FakeChatModel(responses=[AIMessage(content="inner answer")]),
        [],
        name="inner-agent",
    )
    nested_model = FakeChatModel(
        responses=[AIMessage(content="nested answer")], model_name="tool-model"
    )
    retriever = FakeRetriever(
        model=FakeChatModel(
            responses=[AIMessage(content="retrieved answer")],
            model_name="retrieval-model",
        )
    )

    @tool
    def nested_calculation() -> str:
        """Return a nested calculation."""
        return "42"

    def delegate() -> str:
        nested_model.invoke("hi")
        nested_calculation.invoke({})
        retriever.invoke("hi")
        inner.invoke(inputs)
        return "42"

    async def adelegate() -> str:
        await nested_model.ainvoke("hi")
        await nested_calculation.ainvoke({})
        await retriever.ainvoke("hi")
        await inner.ainvoke(inputs)
        return "42"

    dispatch = StructuredTool.from_function(
        func=delegate, coroutine=adelegate, description="Delegate the work."
    )
    await _invoke(
        _agent_with_tool(dispatch, "outer-agent"), inputs, asynchronous
    )

    dispatch_span = finished_span(span_exporter, "execute_tool delegate")
    retrieval_span = finished_span(span_exporter, "retrieval")
    for name, parent in (
        ("invoke_agent inner-agent", dispatch_span),
        ("chat tool-model", dispatch_span),
        ("execute_tool nested_calculation", dispatch_span),
        ("retrieval", dispatch_span),
        ("chat retrieval-model", retrieval_span),
    ):
        assert finished_span(span_exporter, name).parent == parent.context
    _assert_agent_counts(
        metric_reader, {"outer-agent": (2, 1), "inner-agent": (1, 0)}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["sync-close", "async-close", "cancel"])
async def test_native_agent_partial_call_counts(
    ending: str,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    instrument_langchain: None,
) -> None:
    started = asyncio.Event()

    def calculate() -> str:
        started.set()
        return "42"

    async def acalculate() -> str:
        started.set()
        await asyncio.sleep(60)
        return "42"

    calculation = StructuredTool.from_function(
        func=calculate,
        coroutine=acalculate,
        description="Calculate an answer.",
    )
    agent = _agent_with_tool(calculation, "partial-agent")
    inputs = {"messages": [("user", "calculate")]}
    if ending == "sync-close":
        stream = agent.stream(inputs, stream_mode="updates")
        try:
            assert next(stream)
        finally:
            stream.close()
    elif ending == "async-close":
        stream = agent.astream(inputs, stream_mode="updates")
        try:
            assert await anext(stream)
        finally:
            await stream.aclose()
    else:

        async def consume() -> None:
            async for _ in agent.astream(inputs):
                pass

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 5)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

    assert started.is_set() == (ending == "cancel")
    span = finished_span(span_exporter, "invoke_agent partial-agent")
    if ending == "cancel":
        assert (
            span.attributes[error_attributes.ERROR_TYPE]
            == "asyncio.exceptions.CancelledError"
        )
    _assert_agent_counts(
        metric_reader, {"partial-agent": (1, int(ending == "cancel"))}
    )
