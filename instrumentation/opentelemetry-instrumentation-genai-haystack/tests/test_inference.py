# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for classified ``GENERATOR`` components -> ``InferenceInvocation``.

Components are called directly (not through a ``Pipeline``) to isolate the
component-level wrapping from ``Pipeline.run``'s workflow span — the two are
covered together in ``test_workflow.py``.
"""

import json

import pytest
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage
from haystack.utils import Secret

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .test_utils import (
    assert_chat_span_attributes,
    load_messages_attribute,
    message_part_types,
    text_content,
)

OPENAI = GenAIAttributes.GenAiProviderNameValues.OPENAI.value


@pytest.mark.vcr
def test_chat_generator_sync(span_exporter, instrument_with_content):
    generator = OpenAIChatGenerator(model="gpt-4o")
    messages = [
        ChatMessage.from_system("Answer user questions succinctly"),
        ChatMessage.from_assistant("What can I help you with?"),
        ChatMessage.from_user(
            "Who won the World Cup in 2022? Answer in one word."
        ),
    ]
    response = generator.run(messages=messages)
    assert response["replies"][0].text == "Argentina."

    (span,) = span_exporter.get_finished_spans()
    assert_chat_span_attributes(
        span,
        request_model="gpt-4o",
        provider=OPENAI,
        response_model="gpt-4o-2024-05-13",
        input_tokens=42,
        output_tokens=2,
        finish_reasons=("stop",),
    )

    input_messages = load_messages_attribute(
        span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert [message["role"] for message in input_messages] == [
        "system",
        "assistant",
        "user",
    ]
    assert (
        text_content(input_messages[2])
        == "Who won the World Cup in 2022? Answer in one word."
    )

    output_messages = load_messages_attribute(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert text_content(output_messages[0]) == "Argentina."
    assert output_messages[0]["finish_reason"] == "stop"


@pytest.mark.vcr
async def test_chat_generator_async(span_exporter, instrument_with_content):
    generator = OpenAIChatGenerator(model="gpt-4o")
    messages = [
        ChatMessage.from_system("Answer user questions succinctly"),
        ChatMessage.from_assistant("What can I help you with?"),
        ChatMessage.from_user(
            "Who won the World Cup in 2022? Answer in one word."
        ),
    ]
    response = await generator.run_async(messages=messages)
    assert response["replies"][0].text == "Argentina."

    (span,) = span_exporter.get_finished_spans()
    assert_chat_span_attributes(
        span,
        request_model="gpt-4o",
        provider=OPENAI,
        response_model="gpt-4o-2024-08-06",
        input_tokens=42,
        output_tokens=2,
    )


@pytest.mark.vcr
def test_chat_generator_no_content_capture(
    span_exporter, instrument_no_content
):
    generator = OpenAIChatGenerator(model="gpt-4o")
    messages = [
        ChatMessage.from_system("Answer user questions succinctly"),
        ChatMessage.from_assistant("What can I help you with?"),
        ChatMessage.from_user(
            "Who won the World Cup in 2022? Answer in one word."
        ),
    ]
    generator.run(messages=messages)

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
    # Non-content attributes are still recorded.
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"
    assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 42


@pytest.mark.vcr
def test_chat_generator_error(span_exporter, instrument_with_content):
    generator = OpenAIChatGenerator(
        model="gpt-4o", api_key=Secret.from_token("sk-invalid")
    )
    with pytest.raises(Exception) as excinfo:
        generator.run(
            messages=[
                ChatMessage.from_user(
                    "Who won the World Cup in 2022? Answer in one word."
                )
            ]
        )

    err_name_fq = (
        f"{type(excinfo.value).__module__}.{type(excinfo.value).__name__}"
    )
    err_name_short = type(excinfo.value).__name__

    (span,) = span_exporter.get_finished_spans()
    assert not span.status.is_ok
    attributes = span.attributes or {}
    assert attributes[ErrorAttributes.ERROR_TYPE] in (
        err_name_fq,
        err_name_short,
    )
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o"


@pytest.mark.skip(reason="Missing VCR cassette from upstream PR")
@pytest.mark.vcr
def test_tool_calling_captures_tool_call_on_output_message(
    span_exporter, instrument_with_content
):
    generator = OpenAIChatGenerator(model="gpt-4o")
    response = generator.run(
        messages=[ChatMessage.from_user("What is the weather in Berlin")],
        generation_kwargs={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state, e.g. San Francisco, CA",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                },
            ]
        },
    )
    reply = response["replies"][0]
    assert reply.tool_calls[0].tool_name == "get_current_weather"

    (span,) = span_exporter.get_finished_spans()
    assert_chat_span_attributes(
        span,
        request_model="gpt-4o",
        provider=OPENAI,
        response_model="gpt-4o-2024-05-13",
        input_tokens=63,
        output_tokens=15,
        finish_reasons=("tool_calls",),
    )

    output_messages = load_messages_attribute(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert "tool_call" in message_part_types(output_messages[0])
    tool_call_part = next(
        part
        for part in output_messages[0]["parts"]
        if part["type"] == "tool_call"
    )
    assert tool_call_part["name"] == "get_current_weather"
    arguments = tool_call_part["arguments"]
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    assert arguments == {"location": "Berlin"}
