# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``Agent`` component -> ``AgentInvocation``.

Drives a real ``haystack.components.agents.agent.Agent`` end to end against
a fake, local, deterministic ``ChatGenerator`` and a real ``Tool`` -- no
network calls needed. The Agent's own tool-calling turn also exercises the
already-covered ``chat`` (``test_inference.py``) and ``execute_tool``
(``test_tool.py``) code paths as nested spans; this file only asserts on
the outer ``invoke_agent`` span and the overall span tree.
"""

from typing import List

from haystack import component
from haystack.components.agents.agent import Agent
from haystack.dataclasses.chat_message import ChatMessage, ToolCall
from haystack.tools import Tool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .test_utils import load_messages_attribute, message_part_types


@component
class _ScriptedChatGenerator:
    """Returns a tool call on the first turn, a final answer on the second."""

    def __init__(self):
        self.model = "scripted-model"
        self._call_count = 0

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages, tools=None, generation_kwargs=None, **kwargs):
        self._call_count += 1
        if self._call_count == 1:
            reply = ChatMessage.from_assistant(
                text=None,
                tool_calls=[
                    ToolCall(
                        tool_name="get_weather",
                        arguments={"city": "Berlin"},
                        id="call_1",
                    )
                ],
                meta={
                    "model": "scripted-model",
                    "finish_reason": "tool_calls",
                },
            )
        else:
            reply = ChatMessage.from_assistant(
                text="It is sunny in Berlin.",
                meta={"model": "scripted-model", "finish_reason": "stop"},
            )
        return {"replies": [reply]}


def _get_weather(city: str) -> str:
    return f"sunny in {city}"


def _build_agent() -> Agent:
    tool = Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        function=_get_weather,
    )
    return Agent(chat_generator=_ScriptedChatGenerator(), tools=[tool])


def test_agent_run_produces_nested_chat_tool_and_agent_spans(
    span_exporter, instrument_with_content
):
    agent = _build_agent()
    result = agent.run(
        messages=[ChatMessage.from_user("What's the weather in Berlin?")]
    )
    assert result["last_message"].text == "It is sunny in Berlin."

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "chat scripted-model",
        "execute_tool get_weather",
        "chat scripted-model",
        "invoke_agent Agent",
    ]
    chat_span_1, tool_span, chat_span_2, agent_span = spans

    # All three inner spans nest directly under the agent span.
    for inner in (chat_span_1, tool_span, chat_span_2):
        assert inner.parent is not None
        assert inner.parent.span_id == agent_span.context.span_id

    assert agent_span.status.is_ok
    agent_attributes = agent_span.attributes or {}
    assert (
        agent_attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == "invoke_agent"
    )
    assert agent_attributes[GenAIAttributes.GEN_AI_AGENT_NAME] == "Agent"

    input_messages = load_messages_attribute(
        agent_span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert len(input_messages) == 1
    assert input_messages[0]["role"] == "user"

    # Output messages are only the newly generated ones -- the tool call,
    # the tool result, and the final answer -- not the echoed-back input.
    output_messages = load_messages_attribute(
        agent_span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert len(output_messages) == 3
    assert "tool_call" in message_part_types(output_messages[0])
    assert "tool_call_response" in message_part_types(output_messages[1])
    assert output_messages[2]["role"] == "assistant"


def test_agent_run_no_content_capture(span_exporter, instrument_no_content):
    agent = _build_agent()
    agent.run(
        messages=[ChatMessage.from_user("What's the weather in Berlin?")]
    )

    agent_span = next(
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "invoke_agent Agent"
    )
    attributes = agent_span.attributes or {}
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "invoke_agent"


def test_component_defined_after_instrument_is_still_wrapped(
    span_exporter, instrument_with_content
):
    """Regression test: a component class defined *after* ``instrument()``
    runs (the common "instrument first, build pipeline second" ordering)
    must still be classified and wrapped -- not just ones already imported
    at instrumentation time. See MIGRATION_REPORT.md and __init__.py's
    ``_Component._component`` registration hook.
    """

    @component
    class _LateGenerator:
        @component.output_types(replies=List[ChatMessage])
        def run(self, messages, **kwargs):
            return {
                "replies": [
                    ChatMessage.from_assistant(
                        "late", meta={"finish_reason": "stop"}
                    )
                ]
            }

    _LateGenerator().run(messages=[ChatMessage.from_user("hi")])

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "chat"
    assert (span.attributes or {})[
        GenAIAttributes.GEN_AI_OPERATION_NAME
    ] == "chat"
