# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic message parameter extraction."""

import pytest

from opentelemetry.instrumentation.genai.anthropic.messages_extractors import (
    extract_params,
    get_tool_definitions,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
)


def test_extract_params_reads_sampling_params_from_extra_body():
    params = extract_params(
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    )

    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 40


def test_extract_params_prefers_named_sampling_params():
    params = extract_params(
        temperature=0.2,
        top_p=0.3,
        top_k=10,
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    )

    assert params.temperature == 0.2
    assert params.top_p == 0.3
    assert params.top_k == 10


def test_extract_params_ignores_non_numeric_extra_body_sampling_params():
    params = extract_params(
        extra_body={"temperature": "high", "top_p": True, "top_k": 2.5}
    )

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_extract_params_ignores_non_mapping_extra_body():
    params = extract_params(extra_body="temperature=0.7")

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_extract_params_keeps_tools():
    tools = [{"name": "get_weather", "input_schema": {"type": "object"}}]

    params = extract_params(tools=tools)

    assert params.tools is tools


def test_get_tool_definitions_maps_custom_tool_to_function():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    }

    definitions = get_tool_definitions(
        [
            {
                "name": "get_weather",
                "description": "Get weather by city",
                "input_schema": schema,
            }
        ]
    )

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description="Get weather by city",
            parameters=schema,
        )
    ]
    assert definitions[0].type == "function"


def test_get_tool_definitions_maps_explicit_custom_type_to_function():
    definitions = get_tool_definitions([{"type": "custom", "name": "noop"}])

    assert definitions == [
        FunctionToolDefinition(name="noop", description=None, parameters=None)
    ]


def test_get_tool_definitions_maps_server_tool_to_generic():
    definitions = get_tool_definitions(
        [
            {
                "name": "web_search",
                "type": "web_search_20250305",
                "max_uses": 3,
            }
        ]
    )

    assert definitions == [
        GenericToolDefinition(name="web_search", type="web_search_20250305")
    ]


def test_get_tool_definitions_falls_back_to_type_for_unnamed_toolset():
    definitions = get_tool_definitions(
        [{"type": "computer_toolset_20260801", "configs": []}]
    )

    assert definitions == [
        GenericToolDefinition(
            name="computer_toolset_20260801",
            type="computer_toolset_20260801",
        )
    ]


def test_get_tool_definitions_reads_tool_objects():
    class _Tool:
        name = "get_weather"
        description = "Get weather by city"
        input_schema = {"type": "object"}

    definitions = get_tool_definitions([_Tool()])

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description="Get weather by city",
            parameters={"type": "object"},
        )
    ]


def test_get_tool_definitions_mixes_custom_and_server_tools():
    definitions = get_tool_definitions(
        [
            {"name": "get_weather", "input_schema": {"type": "object"}},
            {"name": "bash", "type": "bash_20250124"},
        ]
    )

    assert definitions == [
        FunctionToolDefinition(
            name="get_weather",
            description=None,
            parameters={"type": "object"},
        ),
        GenericToolDefinition(name="bash", type="bash_20250124"),
    ]


@pytest.mark.parametrize("tools", [None, []])
def test_get_tool_definitions_without_tools(tools):
    assert get_tool_definitions(tools) is None
