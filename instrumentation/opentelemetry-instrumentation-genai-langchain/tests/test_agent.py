# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent telemetry exercised through LangChain's public APIs, without network calls."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, create_react_agent
from tests._api import FakeChatModel
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
    (agent_span,) = [
        span
        for span in spans
        if span.parent is None
        and span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == "invoke_agent"
    ]
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
    data = metric_reader.get_metrics_data()
    assert data is not None
    metrics = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    for name, total in (
        ("gen_ai.invoke_agent.inference_calls", inference_calls),
        (
            "gen_ai.invoke_agent.tool_calls",
            int(use_tool and failure != "model"),
        ),
    ):
        (point,) = metrics[name].data.data_points
        assert point.count == 1
        assert type(point.sum) is int
        assert point.sum == total
