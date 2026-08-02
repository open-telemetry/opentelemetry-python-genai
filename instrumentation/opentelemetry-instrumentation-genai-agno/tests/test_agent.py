# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Agent instrumentation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def test_agent_run_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run emits an invoke_agent span."""
    agent = Agent(name="test-sync-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Hello back!")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run("hello world")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-sync-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-sync-agent"
    )


def test_agent_arun_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun emits an invoke_agent span."""
    agent = Agent(name="test-async-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Async hello back!")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("hello async world")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-async-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-async-agent"
    )


def test_tool_call_execute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.execute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-123",
    )
    res = func_call.execute()
    assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-123"
    )


def test_tool_call_aexecute_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that FunctionCall.aexecute emits an execute_tool span."""

    def sample_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    func = Function.from_callable(sample_tool)
    func_call = FunctionCall(
        function=func,
        arguments={"x": 5},
        call_id="call-456",
    )

    async def _run_async() -> None:
        res = await func_call.aexecute()
        assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "execute_tool sample_tool"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "execute_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_NAME) == "sample_tool"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID) == "call-456"
    )
