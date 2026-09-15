# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from crewai import Task
from crewai.lite_agent_output import LiteAgentOutput
from crewai.tools import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from pydantic import BaseModel

from opentelemetry.instrumentation.genai.crewai._messages import (
    agent_tool_definitions,
    crewai_tools,
    messages_to_input_messages,
    output_to_output_messages,
    structured_tool_input_to_arguments,
    task_to_input_messages,
    tool_args_to_arguments,
    tool_description,
    tool_result_to_result,
)

from .test_operations import AddTool


class MixedArgsTool(BaseTool):
    name: str = "mixed"
    description: str = "Mixed arguments"

    def _run(self, first: int, /, second: int = 2, *, third: int = 3) -> int:
        return first + second + third


def test_agent_message_conversion() -> None:
    task = Task(
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
    assert messages[0].name is None
    assert messages[1].name == "helper"

    output = output_to_output_messages(
        LiteAgentOutput(raw="Finished", agent_role="Researcher")
    )
    assert output[0].role == "assistant"
    assert output[0].parts[0].content == "Finished"


def test_agent_message_conversion_handles_odd_inputs() -> None:
    assert messages_to_input_messages("Plain prompt")[0].parts[0].content == (
        "Plain prompt"
    )
    assert (
        messages_to_input_messages([{"role": "user", "content": None}]) == []
    )
    parts_message = messages_to_input_messages(
        [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    )
    assert (
        parts_message[0].parts[0].content == '[{"type": "text", "text": "Hi"}]'
    )
    assert output_to_output_messages(None) == []
    assert output_to_output_messages("plain")[0].parts[0].content == "plain"

    class Structured(BaseModel):
        answer: str

    output = output_to_output_messages(Structured(answer="ok"))
    assert output[0].parts[0].content == '{"answer": "ok"}'


def test_tool_conversion_preserves_arguments() -> None:
    tool = MixedArgsTool()
    assert tool_args_to_arguments(tool, (1, 4), {"third": 6}) == {
        "first": 1,
        "second": 4,
        "third": 6,
    }
    definitions = agent_tool_definitions([tool])
    assert definitions is not None
    assert definitions[0].name == "mixed"
    assert definitions[0].description == "Mixed arguments"


def test_tool_definitions_include_schema_parameters() -> None:
    definitions = agent_tool_definitions([AddTool()])
    assert definitions is not None
    assert definitions[0].parameters is not None
    assert set(definitions[0].parameters["properties"]) == {"left", "right"}
    assert agent_tool_definitions(None) is None
    assert agent_tool_definitions([]) == []


def test_crewai_tools_narrowing() -> None:
    tool = AddTool()
    structured = tool.to_structured_tool()
    assert crewai_tools(None) is None
    assert crewai_tools("not tools") is None
    assert crewai_tools([tool, "junk", structured]) == [tool, structured]
    assert crewai_tools([]) == []


def test_tool_arguments_fall_back_to_envelope() -> None:
    # ``run(*args, **kwargs)`` that does not bind to ``_run`` keeps both.
    assert tool_args_to_arguments(AddTool(), (1, 2, 3), {"key": "value"}) == {
        "args": [1, 2, 3],
        "kwargs": {"key": "value"},
    }


def test_structured_tool_conversion_excludes_config() -> None:
    def invoke(input, config=None, **kwargs):
        return input, config, kwargs

    assert structured_tool_input_to_arguments(
        invoke,
        ({"city": "Paris"},),
        {"config": {"callbacks": []}, "units": "metric"},
    ) == {"city": "Paris", "units": "metric"}


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
    structured = CrewStructuredTool(
        name="add",
        description=composite,
        args_schema=AddTool().args_schema,
        func=lambda **kwargs: None,
    )
    assert tool_description(structured) == "Add two integers"
    assert tool_description(AddTool()) == "Add two integers"
