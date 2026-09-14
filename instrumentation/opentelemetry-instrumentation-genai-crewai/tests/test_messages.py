# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from opentelemetry.instrumentation.genai.crewai._messages import (
    agent_tool_definitions,
    messages_to_input_messages,
    output_to_output_messages,
    structured_tool_input_to_arguments,
    task_to_input_messages,
    tool_args_to_arguments,
    tool_description,
    tool_result_to_result,
)


def test_agent_message_conversion() -> None:
    task = SimpleNamespace(
        description="Research telemetry",
        expected_output="A concise summary",
    )
    task_messages = task_to_input_messages(task, context="Earlier findings")
    assert [part.content for part in task_messages[0].parts] == [
        "Research telemetry",
        "Expected output: A concise summary",
        "Context: Earlier findings",
    ]

    messages = messages_to_input_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi", "name": "helper"},
        ]
    )
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].name == "helper"

    output = output_to_output_messages(SimpleNamespace(raw="Finished"))
    assert output[0].role == "assistant"
    assert output[0].parts[0].content == "Finished"


def test_tool_conversion_preserves_arguments() -> None:
    class Tool:
        name = "mixed"
        description = "Mixed arguments"
        args_schema = None

        def _run(self, first: int, /, second: int = 2, *, third: int = 3):
            return first + second + third

    tool = Tool()
    assert tool_args_to_arguments(tool, (1, 4), {"third": 6}) == {
        "first": 1,
        "second": 4,
        "third": 6,
    }
    definitions = agent_tool_definitions([tool])
    assert definitions is not None
    assert definitions[0].name == "mixed"
    assert definitions[0].description == "Mixed arguments"


def test_structured_tool_conversion_excludes_config() -> None:
    def invoke(input, config=None, **kwargs):
        return input, config, kwargs

    assert structured_tool_input_to_arguments(
        invoke,
        ({"city": "Paris"},),
        {"config": {"callbacks": []}, "units": "metric"},
    ) == {"city": "Paris", "units": "metric"}


def test_agent_message_conversion_handles_odd_inputs() -> None:
    assert task_to_input_messages(None) == []
    assert messages_to_input_messages(42) == []
    assert messages_to_input_messages("Plain prompt")[0].parts[0].content == (
        "Plain prompt"
    )
    assert messages_to_input_messages([{"role": "user"}]) == []
    assert output_to_output_messages(None) == []

    class Structured(BaseModel):
        answer: str

    output = output_to_output_messages(Structured(answer="ok"))
    assert output[0].parts[0].content == '{"answer": "ok"}'


def test_tool_definitions_include_schema_parameters() -> None:
    class Args(BaseModel):
        city: str

    class Tool:
        name = "weather"
        description = "Look up weather"
        args_schema = Args

    definitions = agent_tool_definitions([Tool()])
    assert definitions is not None
    assert definitions[0].parameters is not None
    assert set(definitions[0].parameters["properties"]) == {"city"}
    assert agent_tool_definitions(None) is None
    assert agent_tool_definitions([]) == []


def test_tool_arguments_fall_back_to_envelope() -> None:
    class Tool:
        name = "opaque"
        _run = None

    assert tool_args_to_arguments(Tool(), (1,), {"key": "value"}) == {
        "args": [1],
        "kwargs": {"key": "value"},
    }


def test_structured_tool_conversion_parses_string_input() -> None:
    def invoke(input, config=None, **kwargs):
        return input, config, kwargs

    assert structured_tool_input_to_arguments(
        invoke, ('{"city": "Paris"}',), {}
    ) == {"city": "Paris"}
    assert structured_tool_input_to_arguments(
        invoke, ({"city": "Paris", "security_context": {"x": 1}},), {}
    ) == {"city": "Paris"}
    assert structured_tool_input_to_arguments(invoke, ("not json",), {}) == (
        "not json"
    )
    assert structured_tool_input_to_arguments(
        invoke, ("not json",), {"units": "metric"}
    ) == {"input": "not json", "kwargs": {"units": "metric"}}


def test_tool_result_conversion() -> None:
    assert tool_result_to_result(6) == 6
    assert tool_result_to_result({"a": (1, 2)}) == {"a": [1, 2]}
    assert tool_result_to_result(object) == str(object)


def test_tool_description_strips_composite_prefix() -> None:
    composite = (
        "Tool Name: add\n"
        'Tool Arguments: {"properties": {"left": {"type": "integer"}}}\n'
        "Tool Description: Add two integers"
    )
    assert tool_description(SimpleNamespace(description=composite)) == (
        "Add two integers"
    )
    assert tool_description(SimpleNamespace(description="Plain")) == "Plain"
    assert tool_description(SimpleNamespace(description=None)) is None
    assert tool_description(object()) is None
