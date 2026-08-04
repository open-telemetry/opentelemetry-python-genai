# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agent._call_tool() instrumentation (execute_tool spans)."""
# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest
from qwen_agent.agent import Agent
from qwen_agent.llm.schema import FunctionCall, Message
from qwen_agent.tools.base import ToolServiceError

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import StatusCode


class _StubAgent(Agent):
    """A minimal Agent subclass that skips the heavy Agent.__init__."""

    @classmethod
    def create(cls, name="ToolAgent"):
        obj = cls.__new__(cls)
        obj.name = name
        obj.description = ""
        obj.system_message = ""
        obj.llm = None
        obj.function_map = {}
        obj.extra_generate_cfg = {}
        return obj

    def _run(self, messages, **kwargs):
        raise NotImplementedError


def _make_agent_with_tool(tool_name="get_weather"):
    agent = _StubAgent.create()
    tool = MagicMock()
    tool.description = "Get weather information"
    tool.call = MagicMock(return_value="Sunny, 25 degrees")
    agent.function_map = {tool_name: tool}
    return agent, tool


def _get_tool_spans(span_exporter):
    return [
        s
        for s in span_exporter.get_finished_spans()
        if s.name.startswith("execute_tool")
    ]


def test_call_tool_span_attributes(span_exporter, instrument_with_content):
    """_call_tool() records an execute_tool span with tool attributes."""
    agent, _ = _make_agent_with_tool("get_weather")

    result = agent._call_tool("get_weather", '{"city": "Beijing"}')
    assert result == "Sunny, 25 degrees"

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.name == "execute_tool get_weather"
    assert span.status.status_code != StatusCode.ERROR

    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "get_weather"
    assert attrs[GenAIAttributes.GEN_AI_TOOL_TYPE] == "function"
    assert (
        attrs[GenAIAttributes.GEN_AI_TOOL_DESCRIPTION]
        == "Get weather information"
    )
    assert (
        attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        == '{"city": "Beijing"}'
    )
    assert (
        attrs[GenAIAttributes.GEN_AI_TOOL_CALL_RESULT] == "Sunny, 25 degrees"
    )


def test_call_tool_no_content(span_exporter, instrument_no_content):
    """Without content capture, tool arguments/result stay off the span."""
    agent, _ = _make_agent_with_tool("get_weather")

    agent._call_tool("get_weather", '{"city": "Beijing"}')

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    attrs = dict(tool_spans[0].attributes or {})
    assert GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS not in attrs
    assert GenAIAttributes.GEN_AI_TOOL_CALL_RESULT not in attrs
    assert attrs[GenAIAttributes.GEN_AI_TOOL_NAME] == "get_weather"


def test_call_tool_records_tool_call_id(span_exporter, instrument_no_content):
    """The pending function_id from the message history becomes
    gen_ai.tool.call.id."""
    agent, _ = _make_agent_with_tool("get_weather")
    messages = [
        Message(role="user", content="Weather in Beijing?"),
        Message(
            role="assistant",
            content="",
            function_call=FunctionCall(
                name="get_weather", arguments='{"city": "Beijing"}'
            ),
            extra={"function_id": "call_abc123"},
        ),
    ]

    agent._call_tool("get_weather", '{"city": "Beijing"}', messages=messages)

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    attrs = dict(tool_spans[0].attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ID] == "call_abc123"


def test_call_tool_with_dict_args(span_exporter, instrument_with_content):
    """Dict tool arguments are serialized to JSON on the span."""
    agent, tool = _make_agent_with_tool("search")
    tool.call = MagicMock(return_value="Found 3 results")

    agent._call_tool("search", {"query": "OpenTelemetry"})

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    attrs = dict(tool_spans[0].attributes or {})
    assert (
        attrs[GenAIAttributes.GEN_AI_TOOL_CALL_ARGUMENTS]
        == '{"query":"OpenTelemetry"}'
    )


def test_call_tool_swallowed_error_still_records_span(
    span_exporter, instrument_no_content
):
    """qwen-agent catches generic tool errors and returns an error string;
    the execute_tool span still finalizes without error status."""
    agent, tool = _make_agent_with_tool("broken_tool")
    tool.call = MagicMock(side_effect=RuntimeError("Tool crashed"))

    result = agent._call_tool("broken_tool", "{}")
    assert isinstance(result, str)

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    assert tool_spans[0].status.status_code != StatusCode.ERROR


def test_call_tool_error_re_raises(span_exporter, instrument_no_content):
    """ToolServiceError re-raises unchanged and fails the execute_tool span."""
    agent, tool = _make_agent_with_tool("service_tool")
    tool.call = MagicMock(
        side_effect=ToolServiceError(message="service unavailable")
    )

    with pytest.raises(ToolServiceError):
        agent._call_tool("service_tool", "{}")

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    span = tool_spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = dict(span.attributes or {})
    # Non-builtin exceptions are recorded with their fully qualified name.
    assert (
        attrs[ErrorAttributes.ERROR_TYPE]
        == "qwen_agent.tools.base.ToolServiceError"
    )


def test_call_unknown_tool_records_span(span_exporter, instrument_no_content):
    """Calling an unregistered tool still records an execute_tool span."""
    agent = _StubAgent.create()

    result = agent._call_tool("nonexistent", "{}")
    assert isinstance(result, str)

    tool_spans = _get_tool_spans(span_exporter)
    assert len(tool_spans) == 1
    assert tool_spans[0].name == "execute_tool nonexistent"
