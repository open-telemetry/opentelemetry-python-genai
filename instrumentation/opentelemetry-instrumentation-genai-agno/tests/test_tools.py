# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno tool definitions and instrumentation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.tools import Toolkit
from agno.tools.function import Function
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno.utils import (
    prepare_tool_definitions,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import FunctionToolDefinition


def test_prepare_tool_definitions_returns_none_for_empty() -> None:
    assert prepare_tool_definitions([]) is None
    assert prepare_tool_definitions(None) is None


def test_prepare_tool_definitions_dict_tools() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        },
        {
            "name": "get_time",
            "description": "Get current time",
            "parameters": {"type": "object"},
        },
    ]
    result = prepare_tool_definitions(tools)
    assert result is not None
    assert len(result) == 2

    first = result[0]
    assert isinstance(first, FunctionToolDefinition)
    assert first.name == "get_weather"
    assert first.description == "Get weather for a city"
    assert first.parameters == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }
    assert first.type == "function"

    second = result[1]
    assert isinstance(second, FunctionToolDefinition)
    assert second.name == "get_time"
    assert second.description == "Get current time"
    assert second.parameters == {"type": "object"}


def test_prepare_tool_definitions_callable_tools() -> None:
    def sample_tool(x: int) -> int:
        """Double an integer."""
        return x * 2

    result = prepare_tool_definitions([sample_tool])
    assert result is not None
    assert len(result) == 1
    tool_def = result[0]
    assert isinstance(tool_def, FunctionToolDefinition)
    assert tool_def.name == "sample_tool"
    assert tool_def.description == "Double an integer."
    assert isinstance(tool_def.parameters, dict)
    assert tool_def.parameters.get("type") == "object"
    assert "x" in tool_def.parameters.get("properties", {})


def test_prepare_tool_definitions_function_tools() -> None:
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    func = Function.from_callable(add_numbers)
    result = prepare_tool_definitions([func])
    assert result is not None
    assert len(result) == 1
    tool_def = result[0]
    assert isinstance(tool_def, FunctionToolDefinition)
    assert tool_def.name == "add_numbers"
    assert tool_def.description == "Add two numbers."
    assert isinstance(tool_def.parameters, dict)
    assert tool_def.parameters.get("type") == "object"


def test_prepare_tool_definitions_toolkit_tools() -> None:
    class SampleToolkit(Toolkit):
        def __init__(self) -> None:
            super().__init__(name="sample_toolkit")
            self.register(self.greet)

        def greet(self, name: str) -> str:
            """Greet someone by name."""
            return f"Hello, {name}!"

    result = prepare_tool_definitions([SampleToolkit()])
    assert result is not None
    assert len(result) == 1
    tool_def = result[0]
    assert isinstance(tool_def, FunctionToolDefinition)
    assert tool_def.name == "greet"
    assert tool_def.description == "Greet someone by name."
    assert isinstance(tool_def.parameters, dict)


def test_prepare_tool_definitions_deduplication() -> None:
    def sample_tool(x: int) -> int:
        """Double an integer."""
        return x * 2

    result = prepare_tool_definitions([sample_tool, sample_tool])
    assert result is not None
    assert len(result) == 1
    assert result[0].name == "sample_tool"


def test_agent_run_with_tools(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run emits gen_ai.tool.definitions when tools are present."""

    def sample_tool(location: str) -> str:
        """Get weather for location."""
        return "sunny"

    agent = Agent(
        name="test-tools-sync-agent",
        model=MockModel(id="mock-model"),
        tools=[sample_tool],
    )
    mock_output = ModelResponse(content="The weather is sunny.")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run("what is the weather in Seattle?")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-tools-sync-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-tools-sync-agent"
    )
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span.attributes

    tool_defs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert isinstance(tool_defs, list)
    assert len(tool_defs) == 1
    assert tool_defs[0]["name"] == "sample_tool"
    assert tool_defs[0]["description"] == "Get weather for location."
    assert tool_defs[0]["type"] == "function"
    assert isinstance(tool_defs[0]["parameters"], dict)


def test_agent_arun_with_tools(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.arun emits gen_ai.tool.definitions when tools are present."""

    def sample_tool(location: str) -> str:
        """Get weather for location."""
        return "sunny"

    agent = Agent(
        name="test-tools-async-agent",
        model=MockModel(id="mock-model"),
        tools=[sample_tool],
    )
    mock_output = ModelResponse(content="The weather is sunny.")

    async def _run_async() -> None:
        with patch(
            "agno.models.base.Model.aresponse", return_value=mock_output
        ):
            res = await agent.arun("what is the weather in San Francisco?")
            assert res is not None

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-tools-async-agent"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "invoke_agent"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_NAME)
        == "test-tools-async-agent"
    )
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in span.attributes

    tool_defs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS]
    )
    assert isinstance(tool_defs, list)
    assert len(tool_defs) == 1
    assert tool_defs[0]["name"] == "sample_tool"
    assert tool_defs[0]["description"] == "Get weather for location."
    assert tool_defs[0]["type"] == "function"
    assert isinstance(tool_defs[0]["parameters"], dict)


def test_agent_run_without_tools(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Agent.run does not emit gen_ai.tool.definitions when no tools are present."""
    agent = Agent(name="test-no-tools-agent", model=MockModel(id="mock-model"))
    mock_output = ModelResponse(content="Hello without tools!")

    with (
        patch.object(Agent, "run", wraps=agent.run),
        patch("agno.models.base.Model.response", return_value=mock_output),
    ):
        res = agent.run("hello")
        assert res is not None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "invoke_agent test-no-tools-agent"
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS not in span.attributes
