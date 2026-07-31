# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Messages.parse instrumentation."""

import json

import pytest
from anthropic import NotFoundError
from anthropic.resources.messages import AsyncMessages, Messages
from pydantic import BaseModel

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)

_parse_supported = hasattr(Messages, "parse") and hasattr(
    AsyncMessages, "parse"
)

pytestmark = pytest.mark.skipif(
    not _parse_supported,
    reason="anthropic SDK too old to support Messages.parse",
)


class Greeting(BaseModel):
    greeting: str


def _load_span_messages(span, attribute):
    value = span.attributes.get(attribute)
    assert value is not None
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, list)
    return parsed


def _assert_parsed_response(response) -> None:
    parsed_blocks = [
        getattr(block, "parsed", None) for block in response.content
    ]
    text_blocks = [getattr(block, "text", None) for block in response.content]
    text_payloads = [
        json.loads(text)
        for text in text_blocks
        if isinstance(text, str) and text.startswith("{")
    ]
    greetings = [
        parsed.greeting
        for parsed in parsed_blocks
        if isinstance(parsed, Greeting)
    ]
    greetings.extend(
        parsed.get("greeting")
        for parsed in parsed_blocks
        if isinstance(parsed, dict)
    )
    greetings.extend(
        payload.get("greeting")
        for payload in text_payloads
        if isinstance(payload, dict)
    )

    assert any(greetings), "Expected a parsed greeting in the response content"


def _assert_parse_span(span, *, model: str, response) -> None:
    assert span.name == f"chat {model}"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == (
        GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response.id
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response.model
    )
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes
    assert span.attributes[ServerAttributes.SERVER_ADDRESS] == (
        "api.anthropic.com"
    )


@pytest.mark.vcr()
def test_sync_messages_parse_basic(
    span_exporter, anthropic_client, instrument_no_content
):
    """Messages.parse should emit a chat span for structured output."""
    model = "claude-haiku-4-5"

    response = anthropic_client.messages.parse(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Return JSON with a greeting field set to hello.",
            }
        ],
        output_format=Greeting,
    )

    _assert_parsed_response(response)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_parse_span(spans[0], model=model, response=response)


@pytest.mark.vcr()
def test_sync_messages_parse_captures_content(
    span_exporter, anthropic_client, instrument_with_content
):
    """Messages.parse should capture input and output messages."""
    model = "claude-haiku-4-5"

    anthropic_client.messages.parse(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Return JSON with a greeting field set to hello.",
            }
        ],
        output_format=Greeting,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"]


@pytest.mark.vcr()
def test_sync_messages_parse_api_error(
    span_exporter, anthropic_client, instrument_no_content
):
    """Messages.parse should record API errors."""
    model = "invalid-model-name"

    with pytest.raises(NotFoundError):
        anthropic_client.messages.parse(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            output_format=Greeting,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "NotFoundError" in span.attributes[ErrorAttributes.ERROR_TYPE]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_parse_basic(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """AsyncMessages.parse should emit a chat span for structured output."""
    model = "claude-haiku-4-5"

    response = await async_anthropic_client.messages.parse(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Return JSON with a greeting field set to hello.",
            }
        ],
        output_format=Greeting,
    )

    _assert_parsed_response(response)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_parse_span(spans[0], model=model, response=response)


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_parse_captures_content(
    span_exporter, async_anthropic_client, instrument_with_content
):
    """AsyncMessages.parse should capture input and output messages."""
    model = "claude-haiku-4-5"

    await async_anthropic_client.messages.parse(
        model=model,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": "Return JSON with a greeting field set to hello.",
            }
        ],
        output_format=Greeting,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_parse_api_error(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """AsyncMessages.parse should record API errors."""
    model = "invalid-model-name"

    with pytest.raises(NotFoundError):
        await async_anthropic_client.messages.parse(
            model=model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello"}],
            output_format=Greeting,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "NotFoundError" in span.attributes[ErrorAttributes.ERROR_TYPE]
