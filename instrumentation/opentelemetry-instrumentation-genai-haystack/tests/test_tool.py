# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``haystack.tools.Tool.invoke`` / ``invoke_async`` -> ``ToolInvocation``.

``tool_call_id`` correlation is not populated -- see MIGRATION_REPORT.md.
"""

import json

import pytest
from haystack.tools import Tool
from haystack.tools.errors import ToolInvocationError

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def _get_weather(city: str) -> str:
    return f"sunny in {city}"


def _failing_tool(**kwargs):
    raise RuntimeError("boom")


def _weather_tool() -> Tool:
    return Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=_get_weather,
    )


def test_tool_invoke_sync(span_exporter, instrument_with_content):
    tool = _weather_tool()
    result = tool.invoke(city="Berlin")
    assert result == "sunny in Berlin"

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "execute_tool get_weather"
    attributes = span.attributes or {}
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attributes[GenAIAttributes.GEN_AI_TOOL_NAME] == "get_weather"
    assert attributes[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert (
        attributes[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]
        == "Get the weather for a city"
    )
    assert json.loads(
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    ) == {"city": "Berlin"}
    # A plain string result is a valid AttributeValue on its own and is not
    # JSON-encoded (unlike `arguments`, a dict, which is).
    assert (
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT]
        == "sunny in Berlin"
    )


async def test_tool_invoke_async(span_exporter, instrument_with_content):
    tool = _weather_tool()
    result = await tool.invoke_async(city="Paris")
    assert result == "sunny in Paris"

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "execute_tool get_weather"
    attributes = span.attributes or {}
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert json.loads(
        attributes[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
    ) == {"city": "Paris"}


def test_tool_invoke_no_content_capture(span_exporter, instrument_no_content):
    tool = _weather_tool()
    tool.invoke(city="Berlin")

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attributes
    assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attributes
    assert attributes[GenAIAttributes.GEN_AI_TOOL_NAME] == "get_weather"


def test_tool_invoke_error(span_exporter, instrument_with_content):
    tool = Tool(
        name="failing_tool",
        description="Always fails",
        parameters={"type": "object", "properties": {}},
        function=_failing_tool,
    )
    with pytest.raises(ToolInvocationError):
        tool.invoke()

    (span,) = span_exporter.get_finished_spans()
    assert not span.status.is_ok
    attributes = span.attributes or {}
    assert attributes[ErrorAttributes.ERROR_TYPE] == "ToolInvocationError"
    assert attributes[GenAIAttributes.GEN_AI_TOOL_NAME] == "failing_tool"
