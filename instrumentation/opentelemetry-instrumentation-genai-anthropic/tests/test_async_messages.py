# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for async Messages.create instrumentation."""

import inspect
import json

import pytest
from anthropic import APIConnectionError, AsyncAnthropic, NotFoundError
from anthropic.resources.messages import AsyncMessages as _AsyncMessages

from opentelemetry.instrumentation.genai.anthropic.messages_extractors import (
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
)
from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)

_create_params = set(inspect.signature(_AsyncMessages.create).parameters)
_has_tools_param = "tools" in _create_params
_has_thinking_param = "thinking" in _create_params


def normalize_stop_reason(stop_reason):
    """Map Anthropic stop reasons to GenAI semconv values."""
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason, stop_reason)


def expected_input_tokens(usage):
    """Compute semconv input tokens from Anthropic usage."""
    base = getattr(usage, "input_tokens", 0) or 0
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return base + cache_creation + cache_read


def _load_span_messages(span, attribute):
    value = span.attributes.get(attribute)
    assert value is not None
    assert isinstance(value, str)
    parsed = json.loads(value)
    assert isinstance(parsed, list)
    return parsed


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_basic(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test basic async message creation produces correct span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    response = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "anthropic"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response.id
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response.model
    )
    assert span.attributes[
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ] == expected_input_tokens(response.usage)
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        normalize_stop_reason(response.stop_reason),
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS] == "api.anthropic.com"
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_captures_content(
    span_exporter, async_anthropic_client, instrument_with_content
):
    """Test content capture on async non-streaming create."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
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
    assert output_messages[0]["parts"][0]["type"] == "text"


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_with_all_params(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async message creation with all optional parameters."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello."}]

    await async_anthropic_client.messages.create(
        model=model,
        max_tokens=50,
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        stop_sequences=["STOP"],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == (
        "STOP",
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_token_usage(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test that async token usage is captured correctly."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Count to 5."}]

    response = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes
    assert span.attributes[
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ] == expected_input_tokens(response.usage)
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_stop_reason(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test that async stop reason is captured as finish_reasons array."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hi."}]

    response = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        normalize_stop_reason(response.stop_reason),
    )


@pytest.mark.asyncio
async def test_async_messages_create_connection_error(
    span_exporter, instrument_no_content
):
    """Test that async connection errors are handled correctly."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Hello"}]

    client = AsyncAnthropic(base_url="http://localhost:9999")

    try:
        with pytest.raises(APIConnectionError):
            await client.messages.create(
                model=model,
                max_tokens=100,
                messages=messages,
                timeout=0.1,
            )
    finally:
        await client.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "APIConnectionError" in span.attributes[ErrorAttributes.ERROR_TYPE]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async create(stream=True) returns a wrapped stream and records a span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    response_id = None
    response_model = None
    stop_reason = None
    input_tokens = None
    output_tokens = None

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )
    assert isinstance(stream, AsyncMessagesStreamWrapper)

    async with stream:
        async for chunk in stream:
            if chunk.type == "message_start":
                response_id = chunk.message.id
                response_model = chunk.message.model
                input_tokens = chunk.message.usage.input_tokens
            elif chunk.type == "message_delta":
                stop_reason = chunk.delta.stop_reason
                output_tokens = chunk.usage.output_tokens

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response_model
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == output_tokens
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        normalize_stop_reason(stop_reason),
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming_captures_content(
    span_exporter, async_anthropic_client, instrument_with_content
):
    """Test content capture on async create(stream=True)."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )
    async with stream:
        async for _ in stream:
            pass

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
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming_iteration(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async streaming with direct iteration."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hi."}]

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )

    chunks = []
    async with stream:
        async for chunk in stream:
            chunks.append(chunk)
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming_delegates_response_attribute(
    async_anthropic_client, instrument_no_content
):
    """Async stream wrapper should expose attributes from the wrapped stream."""
    stream = await async_anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hi."}],
        stream=True,
    )

    assert stream.response is not None
    assert stream.response.status_code == 200
    assert stream.response.headers.get("request-id") is not None
    await stream.close()


@pytest.mark.asyncio
async def test_async_messages_create_streaming_connection_error(
    span_exporter, instrument_no_content
):
    """Test that async connection errors during streaming are handled correctly."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Hello"}]

    client = AsyncAnthropic(base_url="http://localhost:9999")

    try:
        with pytest.raises(APIConnectionError):
            await client.messages.create(
                model=model,
                max_tokens=100,
                messages=messages,
                stream=True,
                timeout=0.1,
            )
    finally:
        await client.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "APIConnectionError" in span.attributes[ErrorAttributes.ERROR_TYPE]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.skipif(
    not _has_tools_param,
    reason="anthropic SDK too old to support 'tools' parameter",
)
async def test_async_messages_create_captures_tool_use_content(
    span_exporter, async_anthropic_client, instrument_with_content
):
    """Test that async tool_use blocks are captured as tool_call parts."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "What is the weather in SF?"}]

    await async_anthropic_client.messages.create(
        model=model,
        max_tokens=256,
        messages=messages,
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather by city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "get_weather"},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )

    assert any(
        part.get("type") == "tool_call"
        for message in output_messages
        for part in message.get("parts", [])
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.skipif(
    not _has_thinking_param,
    reason="anthropic SDK too old to support 'thinking' parameter",
)
async def test_async_messages_create_captures_thinking_content(
    span_exporter, async_anthropic_client, instrument_with_content
):
    """Test that async thinking blocks are captured as reasoning parts."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "What is 17*19? Think first."}]

    await async_anthropic_client.messages.create(
        model=model,
        max_tokens=16000,
        messages=messages,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )

    assert any(
        part.get("type") == "reasoning"
        for message in output_messages
        for part in message.get("parts", [])
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_stream_wrapper_finalize_idempotent(
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
):
    """Fully consumed async stream plus explicit close should still yield one span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )

    response_id = None
    response_model = None
    stop_reason = None
    input_tokens = None
    output_tokens = None

    async for chunk in stream:
        if chunk.type == "message_start":
            response_id = chunk.message.id
            response_model = chunk.message.model
            input_tokens = expected_input_tokens(chunk.message.usage)
        elif chunk.type == "message_delta":
            stop_reason = chunk.delta.stop_reason
            output_tokens = chunk.usage.output_tokens
            input_tokens = expected_input_tokens(chunk.usage)

    await stream.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == response_id
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL]
        == response_model
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == output_tokens
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        normalize_stop_reason(stop_reason),
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_aggregates_cache_tokens(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Async non-streaming response with cache tokens aggregates correctly."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    response = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS in span.attributes
    assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS in span.attributes
    assert span.attributes[
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS
    ] == expected_input_tokens(response.usage)
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
    assert (
        span.attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS]
        == cache_creation
    )
    assert span.attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == cache_read


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming_aggregates_cache_tokens(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Async streaming response with cache tokens aggregates correctly."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    input_tokens = None
    output_tokens = None
    cache_creation = None
    cache_read = None

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )
    async with stream:
        async for chunk in stream:
            if chunk.type == "message_delta":
                input_tokens = expected_input_tokens(chunk.usage)
                output_tokens = chunk.usage.output_tokens
                cache_creation = getattr(
                    chunk.usage, "cache_creation_input_tokens", None
                )
                cache_read = getattr(
                    chunk.usage, "cache_read_input_tokens", None
                )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS in span.attributes
    assert GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == output_tokens
    )
    assert (
        span.attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS]
        == cache_creation
    )
    assert span.attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == cache_read


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_stream_propagation_error(
    span_exporter, async_anthropic_client, instrument_no_content, monkeypatch
):
    """Mid-stream async errors must propagate and record error on span."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    stream = await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
        stream=True,
    )

    class ErrorInjectingStreamDelegate:
        def __init__(self, inner):
            self._inner = inner
            self._count = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self._count == 1:
                raise ConnectionError("connection reset during stream")
            self._count += 1
            return await self._inner.__anext__()

        async def close(self):
            return await self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        stream, "stream", ErrorInjectingStreamDelegate(stream.stream)
    )

    with pytest.raises(
        ConnectionError, match="connection reset during stream"
    ):
        async with stream:
            async for _ in stream:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_streaming_user_exception(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test that user raised exceptions are propagated from async streams."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    with pytest.raises(ValueError, match="User raised exception"):
        stream = await async_anthropic_client.messages.create(
            model=model,
            max_tokens=100,
            messages=messages,
            stream=True,
        )
        async with stream:
            async for _ in stream:
                raise ValueError("User raised exception")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_api_error(
    span_exporter, async_anthropic_client, instrument_no_content
):
    """Test async API errors are recorded and re-raised unchanged."""
    model = "invalid-model-name"
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(NotFoundError):
        await async_anthropic_client.messages.create(
            model=model,
            max_tokens=100,
            messages=messages,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert ErrorAttributes.ERROR_TYPE in span.attributes
    assert "NotFoundError" in span.attributes[ErrorAttributes.ERROR_TYPE]


@pytest.mark.asyncio
@pytest.mark.vcr()
async def test_async_messages_create_event_only_no_content_in_span(
    span_exporter,
    log_exporter,
    async_anthropic_client,
    instrument_event_only,
):
    """Test EVENT_ONLY mode emits async create content as a log event only."""
    model = "claude-sonnet-4-20250514"
    messages = [{"role": "user", "content": "Say hello in one word."}]

    await async_anthropic_client.messages.create(
        model=model,
        max_tokens=100,
        messages=messages,
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    log_record = logs[0].log_record
    assert log_record.event_name == "gen_ai.client.inference.operation.details"
    assert log_record.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "anthropic"
