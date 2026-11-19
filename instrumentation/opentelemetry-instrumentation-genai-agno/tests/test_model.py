# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Model inference instrumentation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from agno.agent import Agent
from agno.models.message import Message

try:
    from agno.models.message import MessageMetrics
except ImportError:
    from agno.models.message import Metrics as MessageMetrics

from agno.models.response import ModelResponse
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.instrumentation.genai.agno.utils import (
    resolve_model_provider,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes.error_attributes import (
    ERROR_TYPE,
)
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode


class _SyncModel(MockModel):
    """Model implementation for testing synchronous invocations."""

    def __init__(self, **kwargs: Any) -> None:
        self.temperature = kwargs.pop("temperature", None)
        self.top_p = kwargs.pop("top_p", None)
        self.top_k = kwargs.pop("top_k", None)
        self.max_tokens = kwargs.pop("max_tokens", None)
        self.frequency_penalty = kwargs.pop("frequency_penalty", None)
        self.presence_penalty = kwargs.pop("presence_penalty", None)
        self.seed = kwargs.pop("seed", None)
        self.stop = kwargs.pop("stop", None)
        self.base_url = kwargs.pop("base_url", None)
        super().__init__(**kwargs)

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="Hello from test model!",
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-test-123",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=12,
                output_tokens=7,
                cache_read_tokens=4,
                cache_write_tokens=2,
                reasoning_tokens=1,
            ),
        )


class _AsyncModel(MockModel):
    """Model implementation for testing asynchronous invocations."""

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="Async hello from test model!",
            provider_data={
                "model": "claude-3-5-sonnet",
                "id": "msg-test-456",
                "finish_reason": "end_turn",
            },
            response_usage=MessageMetrics(
                input_tokens=25,
                output_tokens=15,
            ),
        )


class _StreamModel(MockModel):
    """Model implementation for testing stream invocations."""

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponse(content="chunk 1 ")
        yield ModelResponse(
            content="chunk 2",
            provider_data={
                "model": "gpt-4o-stream",
                "id": "stream-resp-789",
                "finish_reason": "stop",
            },
            response_usage=MessageMetrics(
                input_tokens=30,
                output_tokens=10,
            ),
        )

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponse(content="async chunk 1 ")
        yield ModelResponse(
            content="async chunk 2",
            provider_data={
                "model": "claude-stream",
                "id": "async-stream-resp-101",
                "finish_reason": "end_turn",
            },
            response_usage=MessageMetrics(
                input_tokens=40,
                output_tokens=20,
            ),
        )


def test_model_sync_response(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.response emits a chat span with expected attributes."""
    model = _SyncModel(
        id="test-gpt-4o",
        provider="OpenAI",
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        frequency_penalty=0.5,
        presence_penalty=0.2,
        seed=42,
        stop=["END"],
        base_url="https://api.openai.com:8080/v1",
    )
    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Hi there"),
    ]

    response = model.response(messages=messages)
    assert response.content == "Hello from test model!"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat test-gpt-4o"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "test-gpt-4o"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "gpt-4o-2024-08-06"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "chatcmpl-test-123"
    )
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("stop",)
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 12
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 7
    assert (
        span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
        )
        == 4
    )
    assert (
        span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
        )
        == 1
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.7
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.9
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 256
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY)
        == 0.5
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY)
        == 0.2
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_SEED) == 42
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES
    ) == ("END",)
    assert span.attributes.get("server.address") == "api.openai.com"
    assert span.attributes.get("server.port") == 8080

    # Content capture is off by default
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_model_sync_response_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Model.response records messages when content capture is enabled."""
    model = _SyncModel(id="test-gpt-4o", provider="OpenAI")
    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="User prompt"),
    ]

    model.response(messages=messages)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_messages is not None
    assert len(input_messages) == 2
    assert input_messages[0]["role"] == "system"
    assert input_messages[0]["parts"][0]["content"] == "System prompt"
    assert input_messages[1]["role"] == "user"
    assert input_messages[1]["parts"][0]["content"] == "User prompt"

    output_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert len(output_messages) == 1
    assert output_messages[0]["role"] == "assistant"
    assert (
        output_messages[0]["parts"][0]["content"] == "Hello from test model!"
    )


def test_model_async_response(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.aresponse emits a chat span."""
    model = _AsyncModel(id="claude-3-5", provider="Anthropic")
    messages = [Message(role="user", content="Async prompt")]

    async def _run() -> None:
        return await model.aresponse(messages=messages)

    response = asyncio.run(_run())
    assert response.content == "Async hello from test model!"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat claude-3-5"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "anthropic"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "claude-3-5"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "claude-3-5-sonnet"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "msg-test-456"
    )
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("stop",)
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 25
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 15
    )


def test_model_async_response_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Model.aresponse records messages when content capture is enabled."""
    model = _AsyncModel(id="claude-3-5", provider="Anthropic")
    messages = [Message(role="user", content="Async prompt")]

    async def _run() -> None:
        await model.aresponse(messages=messages)

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_messages is not None
    assert len(input_messages) == 1
    assert input_messages[0]["parts"][0]["content"] == "Async prompt"

    output_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert len(output_messages) == 1
    assert (
        output_messages[0]["parts"][0]["content"]
        == "Async hello from test model!"
    )


def test_model_response_stream(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.response_stream emits a streamed chat span."""
    model = _StreamModel(id="stream-model", provider="OpenAI")
    messages = [Message(role="user", content="Stream prompt")]

    chunks = list(model.response_stream(messages=messages))
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat stream-model"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "gpt-4o-stream"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "stream-resp-789"
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 30
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 10
    )
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK in span.attributes
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_model_response_stream_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Model.response_stream concatenates chunks when content capture is enabled."""
    model = _StreamModel(id="stream-model", provider="OpenAI")
    messages = [Message(role="user", content="Stream prompt")]

    list(model.response_stream(messages=messages))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    output_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert len(output_messages) == 1
    assert output_messages[0]["parts"][0]["content"] == "chunk 1 chunk 2"


def test_model_aresponse_stream(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Model.aresponse_stream emits an async streamed chat span."""
    model = _StreamModel(id="async-stream-model", provider="Anthropic")
    messages = [Message(role="user", content="Async stream prompt")]

    async def _run() -> list[Any]:
        results = []
        async for chunk in model.aresponse_stream(messages=messages):
            results.append(chunk)
        return results

    chunks = asyncio.run(_run())
    assert len(chunks) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat async-stream-model"
    assert span.kind == SpanKind.CLIENT
    assert span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "anthropic"
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "claude-stream"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "async-stream-resp-101"
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 40
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 20
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_model_aresponse_stream_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that Model.aresponse_stream concatenates chunks when content capture is enabled."""
    model = _StreamModel(id="async-stream-model", provider="Anthropic")
    messages = [Message(role="user", content="Async stream prompt")]

    async def _run() -> None:
        async for _ in model.aresponse_stream(messages=messages):
            pass

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    output_messages: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages is not None
    assert len(output_messages) == 1
    assert (
        output_messages[0]["parts"][0]["content"]
        == "async chunk 1 async chunk 2"
    )


def test_model_response_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that exceptions during Model.response mark the span with ERROR status and error.type."""

    class ErrorModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> Any:
            raise ValueError("Provider connection failed")

    model = ErrorModel(id="err-model", provider="OpenAI")

    with pytest.raises(ValueError, match="Provider connection failed"):
        model.response(messages=[Message(role="user", content="Hi")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ValueError"


def test_model_aresponse_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that exceptions during Model.aresponse mark the span with ERROR status and error.type."""

    class AsyncErrorModel(MockModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            raise ConnectionResetError("Async connection reset")

    model = AsyncErrorModel(id="async-err-model", provider="Anthropic")

    async def _run() -> None:
        await model.aresponse(messages=[Message(role="user", content="Hi")])

    with pytest.raises(ConnectionResetError, match="Async connection reset"):
        asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ConnectionResetError"


def test_model_stream_error_stream_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that stream-side errors mid-iteration mark the span with ERROR status."""

    class StreamSideErrorModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="ok chunk")
            raise ConnectionError("Connection lost mid-stream")

    model = StreamSideErrorModel(id="stream-side-err-model", provider="OpenAI")

    with pytest.raises(ConnectionError, match="Connection lost mid-stream"):
        for _ in model.response_stream(
            messages=[Message(role="user", content="Hi")]
        ):
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ConnectionError"
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )


def test_model_stream_error_caller_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller-side errors inside context manager mark the span with ERROR status and preserve reported finish reasons."""

    class InfiniteStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="endless",
                provider_data={
                    "id": "resp-caller-1",
                    "finish_reason": "length",
                },
            )
            while True:
                yield ModelResponse(content="endless")

    model = InfiniteStreamModel(id="caller-err-model", provider="OpenAI")

    with pytest.raises(RuntimeError, match="Caller aborted"):
        with model.response_stream(
            messages=[Message(role="user", content="Hi")],
        ) as stream:
            for chunk in stream:
                if chunk.content is not None:
                    raise RuntimeError("Caller aborted")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("length",)


def test_model_response_stream_early_close(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that closing a sync response stream before it is drained finalizes the span once."""
    import gc

    class MultiChunkStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="chunk 1")
            yield ModelResponse(content="chunk 2")
            yield ModelResponse(content="chunk 3")

    model = MultiChunkStreamModel(id="early-close-model", provider="OpenAI")
    stream = model.response_stream(
        messages=[Message(role="user", content="Hi")],
    )
    first = next(c for c in stream if c.content is not None)
    assert first.content == "chunk 1"
    stream.close()
    del stream
    gc.collect()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat early-close-model"
    assert span.status.status_code != StatusCode.ERROR
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )


def test_model_response_stream_early_close_keeps_finish_reason(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that closing a stream early preserves provider finish_reason on stream_data."""

    class _LengthStream(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="partial",
                provider_data={"id": "resp-1", "finish_reason": "length"},
            )
            yield ModelResponse(content="more")

    stream = _LengthStream(id="m", provider="OpenAI").response_stream(
        messages=[Message(role="user", content="hi")],
    )
    next(c for c in stream if c.content is not None)
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("length",)


def test_model_aresponse_stream_error_stream_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that async stream-side errors mid-iteration mark the span with ERROR status."""

    class AsyncStreamSideErrorModel(MockModel):
        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="async ok chunk")
            raise ConnectionError("Async connection lost mid-stream")

    model = AsyncStreamSideErrorModel(
        id="async-stream-side-err-model", provider="Anthropic"
    )

    async def _run() -> None:
        async for _ in model.aresponse_stream(
            messages=[Message(role="user", content="Hi")]
        ):
            pass

    with pytest.raises(
        ConnectionError, match="Async connection lost mid-stream"
    ):
        asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ConnectionError"
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )


def test_model_aresponse_stream_error_caller_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller-side errors inside async context manager mark the span with ERROR status and preserve reported finish reasons."""

    class AsyncInfiniteStreamModel(MockModel):
        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="endless",
                provider_data={
                    "id": "resp-acaller-1",
                    "stop_reason": "max_tokens",
                },
            )
            while True:
                yield ModelResponse(content="endless")

    model = AsyncInfiniteStreamModel(
        id="async-caller-err-model", provider="Anthropic"
    )

    async def _run() -> None:
        async with model.aresponse_stream(
            messages=[Message(role="user", content="Hi")],
        ) as stream:
            async for chunk in stream:
                if chunk.content is not None:
                    raise RuntimeError("Async caller aborted")

    with pytest.raises(RuntimeError, match="Async caller aborted"):
        asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("length",)


def test_model_aresponse_stream_early_close(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that closing an async response stream early finalizes the span once."""
    import gc

    class AsyncMultiChunkStreamModel(MockModel):
        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="async chunk 1")
            yield ModelResponse(content="async chunk 2")

    model = AsyncMultiChunkStreamModel(
        id="async-early-close-model", provider="Anthropic"
    )

    async def _run() -> None:
        stream = model.aresponse_stream(
            messages=[Message(role="user", content="Hi")],
        )
        first = await stream.__anext__()
        while first.content is None:
            first = await stream.__anext__()
        assert first.content == "async chunk 1"
        await stream.aclose()
        del stream
        gc.collect()

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat async-early-close-model"
    assert span.status.status_code != StatusCode.ERROR
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )


def test_model_tool_call_arguments_json_decoding_and_fallback(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that JSON-encoded tool call arguments are decoded while malformed JSON strings are preserved."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    turn = 0

    def _handler(_request: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-tool-args-1",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4o-2024-08-06",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_valid",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "Paris", "unit": "celsius"}',
                                        },
                                    },
                                    {
                                        "id": "call_malformed",
                                        "type": "function",
                                        "function": {
                                            "name": "raw_tool",
                                            "arguments": "{not valid json",
                                        },
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-tool-args-2",
                "object": "chat.completion",
                "created": 1700000001,
                "model": "gpt-4o-2024-08-06",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Completed tool check",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                },
            },
        )

    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    model = OpenAIChat(id="tool-args-model", client=client)
    model.response(
        messages=[
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "hist_call",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query": "otel"}',
                        },
                    }
                ],
            )
        ],
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]

    input_msgs: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_msgs[0]["parts"][0]["arguments"] == {"query": "otel"}

    output_msgs: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    parts = output_msgs[0]["parts"]
    assert parts[0]["name"] == "get_weather"
    assert parts[0]["arguments"] == {"city": "Paris", "unit": "celsius"}
    assert parts[1]["name"] == "raw_tool"
    assert parts[1]["arguments"] == "{not valid json"


def test_agent_multi_turn_tool_calling_spans(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that an Agent tool-calling loop creates parent invoke_agent with child chat and tool spans."""
    call_turn = 0

    class MultiTurnModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            nonlocal call_turn
            call_turn += 1
            if call_turn == 1:
                return ModelResponse(
                    content=None,
                    tool_calls=[
                        {
                            "id": "call_calc",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"x": 6, "y": 7}',
                            },
                        }
                    ],
                    provider_data={"finish_reason": "tool_calls"},
                    response_usage=MessageMetrics(
                        input_tokens=15, output_tokens=8
                    ),
                )
            return ModelResponse(
                content="The result is 42.",
                provider_data={"finish_reason": "stop"},
                response_usage=MessageMetrics(
                    input_tokens=28, output_tokens=12
                ),
            )

    def calculate(x: int, y: int) -> int:
        return x * y

    agent = Agent(
        name="math-agent",
        model=MultiTurnModel(id="multi-turn-model", provider="OpenAI"),
        tools=[calculate],
    )

    result = agent.run("What is 6 times 7?")
    assert "42" in str(result.content)

    spans = span_exporter.get_finished_spans()
    # Expect: 2 chat spans + 1 execute_tool span + 1 invoke_agent span = 4 spans
    assert len(spans) == 4

    agent_spans = [s for s in spans if s.name == "invoke_agent math-agent"]
    chat_spans = [s for s in spans if s.name == "chat multi-turn-model"]
    tool_spans = [s for s in spans if s.name == "execute_tool calculate"]

    assert len(agent_spans) == 1
    assert len(chat_spans) == 2
    assert len(tool_spans) == 1

    agent_span = agent_spans[0]
    agent_span_id = agent_span.context.span_id

    # Verify parent-child hierarchy
    for cs in chat_spans:
        assert cs.parent is not None
        assert cs.parent.span_id == agent_span_id
        assert cs.kind == SpanKind.CLIENT

    tool_span = tool_spans[0]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span_id
    assert tool_span.kind == SpanKind.INTERNAL

    # Verify finish reasons: first turn is tool_call, second turn is stop
    assert chat_spans[0].attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("tool_call",)
    assert chat_spans[1].attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("stop",)


def test_model_uninstrumentation(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that uninstrumenting Agno disables model span emission."""
    model = _SyncModel(id="test-model", provider="OpenAI")
    messages = [Message(role="user", content="Hello")]

    model.response(messages=messages)
    assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    AgnoInstrumentor().uninstrument()

    model.response(messages=messages)
    assert len(span_exporter.get_finished_spans()) == 0


def test_resolve_model_provider() -> None:
    """Test resolve_model_provider resolution logic."""
    assert (
        resolve_model_provider(_SyncModel(id="m", provider="OpenAI"))
        == "openai"
    )
    assert (
        resolve_model_provider(_SyncModel(id="m", provider="Anthropic"))
        == "anthropic"
    )
    assert (
        resolve_model_provider(_SyncModel(id="m", provider="AWS"))
        == "aws.bedrock"
    )
    assert (
        resolve_model_provider(_SyncModel(id="m", provider="groq")) == "groq"
    )
    assert (
        resolve_model_provider(
            _SyncModel(id="m", provider="unknown_custom_xyz")
        )
        == "unknown_custom_xyz"
    )
    assert (
        resolve_model_provider(_SyncModel(id="m", provider=None)) == "unknown"
    )


def test_resolve_model_provider_prefers_class() -> None:
    """Test that class hierarchy mapping takes precedence over ambiguous default provider strings."""
    from dataclasses import dataclass

    @dataclass
    class AzureAIFoundry(MockModel):
        provider: str = "Azure"

    @dataclass
    class WatsonX(MockModel):
        provider: str = "IBM"

    assert (
        resolve_model_provider(AzureAIFoundry(id="m")) == "azure.ai.inference"
    )
    assert resolve_model_provider(WatsonX(id="m")) == "ibm.watsonx.ai"


def test_resolve_model_provider_openai_like() -> None:
    """Test that OpenAILike with inherited default provider resolves to unknown."""
    from dataclasses import dataclass

    try:
        from agno.models.openai.like import OpenAILike

        model = OpenAILike(id="llama3", base_url="http://localhost:8000/v1")
    except ImportError:

        @dataclass
        class OpenAIChat(MockModel):
            provider: str = "OpenAI"

        @dataclass
        class OpenAILike(OpenAIChat):
            pass

        model = OpenAILike(id="llama3")

    assert resolve_model_provider(model) == "unknown"


@pytest.mark.parametrize(
    "raw_reason", ["tool_calls", "tool_use", "function_call"]
)
def test_extract_model_finish_reasons_maps_to_tool_call(
    raw_reason: str,
) -> None:
    """Test that provider tool finish reasons normalize to semconv 'tool_call'."""
    from opentelemetry.instrumentation.genai.agno.utils import (
        extract_model_finish_reasons,
    )

    msg = Message(
        role="assistant", provider_data={"finish_reason": raw_reason}
    )
    assert extract_model_finish_reasons(msg) == ["tool_call"]


def test_model_multimodal_inputs_and_outputs(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that multimodal inputs (images, audio, videos, files) and outputs are captured."""
    from base64 import b64encode

    from agno.media import Audio, File, Image, Video

    raw_audio = b"\x00\x01\x02\x03"
    raw_image = b"\x89PNG\r\n\x1a\n"
    data_url = f"data:image/png;base64,{b64encode(raw_image).decode()}"

    class MultimodalModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                content="Processed all media",
                images=[
                    Image(
                        url="https://example.com/generated.png",
                        mime_type="image/png",
                    )
                ],
                audio=Audio(content=raw_audio, format="mp3"),
                response_usage=MessageMetrics(
                    input_tokens=50, output_tokens=15
                ),
            )

    model = MultimodalModel(id="gpt-4o", provider="OpenAI")
    messages = [
        Message(
            role="user",
            content="Analyze these attachments",
            images=[
                Image(url="https://example.com/photo.jpg", format="jpg"),
                Image(url=data_url),
            ],
            audio=[Audio(content=raw_audio, format="wav")],
            videos=[Video(filepath="/tmp/clip.mp4", format="mp4")],
            files=[
                File(
                    id="file-abc123",
                    mime_type="application/pdf",
                    external={"id": "file-abc123"},
                ),
                File(url="https://example.com/report.pdf"),
            ],
        )
    ]

    model.response(messages=messages)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_msgs: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert len(input_msgs) == 1
    parts = input_msgs[0]["parts"]
    assert parts[0] == {"type": "text", "content": "Analyze these attachments"}
    assert parts[1] == {
        "type": "uri",
        "modality": "image",
        "mime_type": "image/jpeg",
        "uri": "https://example.com/photo.jpg",
    }
    assert parts[2] == {
        "type": "blob",
        "modality": "image",
        "mime_type": "image/png",
        "content": raw_image,
    }
    assert parts[3] == {
        "type": "blob",
        "modality": "audio",
        "mime_type": "audio/wav",
        "content": raw_audio,
    }
    assert parts[4] == {
        "type": "uri",
        "modality": "video",
        "mime_type": "video/mp4",
        "uri": "/tmp/clip.mp4",
    }
    assert parts[5] == {
        "type": "file",
        "modality": "document",
        "mime_type": "application/pdf",
        "file_id": "file-abc123",
    }
    assert parts[6] == {
        "type": "uri",
        "modality": "document",
        "mime_type": "application/pdf",
        "uri": "https://example.com/report.pdf",
    }

    output_msgs: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert len(output_msgs) == 1
    out_parts = output_msgs[0]["parts"]
    assert out_parts[0] == {"type": "text", "content": "Processed all media"}
    assert out_parts[1] == {
        "type": "uri",
        "modality": "image",
        "mime_type": "image/png",
        "uri": "https://example.com/generated.png",
    }
    assert out_parts[2] == {
        "type": "blob",
        "modality": "audio",
        "mime_type": "audio/mpeg",
        "content": raw_audio,
    }


class _NoUsageModel(MockModel):
    """Model implementation returning responses without usage information."""

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="hi from no-usage model")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(content="async hi from no-usage model")

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponse(content="chunk 1 ")
        yield ModelResponse(content="chunk 2")

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        yield ModelResponse(content="async chunk 1 ")
        yield ModelResponse(content="async chunk 2")


def test_model_response_without_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that model response without reported usage emits no usage attributes."""
    model = _NoUsageModel(id="gpt-4o")
    model.response(messages=[Message(role="user", content="hi")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs


def test_model_aresponse_without_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that async model response without reported usage emits no usage attributes."""
    model = _NoUsageModel(id="gpt-4o")
    asyncio.run(model.aresponse(messages=[Message(role="user", content="hi")]))

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs


def test_model_response_stream_without_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that streamed model response without reported usage emits no usage attributes."""
    model = _NoUsageModel(id="gpt-4o")
    for _ in model.response_stream(
        messages=[Message(role="user", content="hi")]
    ):
        pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs


def test_model_aresponse_stream_without_usage(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that async streamed model response without reported usage emits no usage attributes."""
    model = _NoUsageModel(id="gpt-4o")

    async def _run() -> None:
        async for _ in model.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        ):
            pass

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in attrs
    assert GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS not in attrs


def test_model_tool_loop_does_not_reuse_previous_call_data(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that subsequent tool-loop calls do not inherit response_id, usage, finish_reason, or content from earlier calls."""
    turn = 0

    class _Model(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            nonlocal turn
            turn += 1
            if turn == 1:
                return ModelResponse(
                    content="Calling calculate...",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": "{}",
                            },
                        }
                    ],
                    provider_data={
                        "id": "resp-1",
                        "model": "m-snapshot-1",
                        "finish_reason": "tool_calls",
                    },
                    response_usage=MessageMetrics(
                        input_tokens=15, output_tokens=8
                    ),
                )
            return ModelResponse(content=None)

    def calculate() -> int:
        return 42

    Agent(model=_Model(id="m", provider="OpenAI"), tools=[calculate]).run("hi")

    chat_spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "chat m"
    ]
    assert len(chat_spans) == 2

    first, second = chat_spans
    assert first.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID) == "resp-1"
    assert first.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("tool_call",)
    assert (
        first.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 15
    )

    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in second.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in second.attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in second.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in second.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in second.attributes
    )
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in second.attributes


def test_model_async_tool_loop_does_not_reuse_previous_call_data(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test async tool-loop calls do not inherit response_id, usage, finish_reason, or content from earlier calls."""
    turn = 0

    class _AsyncLoopModel(MockModel):
        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            nonlocal turn
            turn += 1
            if turn == 1:
                return ModelResponse(
                    content="Calling calculate...",
                    tool_calls=[
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": "{}",
                            },
                        }
                    ],
                    provider_data={
                        "id": "resp-async-1",
                        "model": "m-async-1",
                        "finish_reason": "tool_use",
                    },
                    response_usage=MessageMetrics(
                        input_tokens=15, output_tokens=8
                    ),
                )
            return ModelResponse(content="done")

    def calculate() -> int:
        return 42

    agent = Agent(
        model=_AsyncLoopModel(id="m", provider="Anthropic"), tools=[calculate]
    )
    asyncio.run(agent.arun("hi"))

    chat_spans = [
        s for s in span_exporter.get_finished_spans() if s.name == "chat m"
    ]
    assert len(chat_spans) == 2

    second = chat_spans[1]
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in second.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in second.attributes
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in second.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in second.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in second.attributes
    )
    output_msgs: Any = second.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert list(output_msgs[0]["parts"]) == [
        {"type": "text", "content": "done"}
    ]


def test_gemini_reasoning_tokens_added_to_output_tokens_and_max_output_tokens(
    instrument_agno,
    span_exporter,
) -> None:
    """Test real Gemini adapter reasoning_tokens, max_output_tokens, nested request_params['config'], and audio tokens."""
    from unittest.mock import MagicMock

    from agno.models.google import Gemini
    from google.genai.types import (
        Candidate,
        Content,
        GenerateContentResponse,
        GenerateContentResponseUsageMetadata,
        Part,
    )

    captured_kwargs: list[dict[str, Any]] = []
    mock_client = MagicMock()

    def _fake_generate_content(**kwargs: Any) -> GenerateContentResponse:
        captured_kwargs.append(kwargs)
        return GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="579")]),
                )
            ],
            usage_metadata=GenerateContentResponseUsageMetadata(
                prompt_token_count=31,
                candidates_token_count=20,
                thoughts_token_count=54,
                total_token_count=105,
            ),
        )

    mock_client.models.generate_content.side_effect = _fake_generate_content

    real_gemini = Gemini(
        id="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=512,
        client=mock_client,
    )
    real_gemini.response(
        messages=[Message(role="user", content="Add 123 and 456")]
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "gcp.gemini"
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.7
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.9
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 512
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 31
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 74
    if GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS in attrs:
        assert (
            attrs.get(GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)
            == 54
        )

    span_exporter.clear()

    override_gemini = Gemini(
        id="gemini-2.5-flash",
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=512,
        request_params={"config": {"temperature": 0.2}},
        client=mock_client,
    )
    override_gemini.response(
        messages=[Message(role="user", content="Nested config override")]
    )

    assert captured_kwargs[-1]["config"] == {"temperature": 0.2}
    (override_span,) = span_exporter.get_finished_spans()
    override_attrs = override_span.attributes
    assert (
        override_attrs.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.2
    )
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_P not in override_attrs
    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS not in override_attrs

    span_exporter.clear()

    class _AudioGeminiLikeModel(MockModel):
        max_output_tokens: int | None = 512

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                content="579",
                response_usage=MessageMetrics(
                    input_tokens=31,
                    output_tokens=20,
                    reasoning_tokens=54,
                    audio_input_tokens=5,
                    audio_output_tokens=10,
                ),
            )

    audio_model = _AudioGeminiLikeModel(
        id="gemini-2.5-flash", provider="Google"
    )
    audio_model.response(
        messages=[Message(role="user", content="Audio token check")]
    )
    (audio_span,) = span_exporter.get_finished_spans()
    assert (
        audio_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
        == 74
    )
    assert (
        audio_span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS
        )
        == 54
    )
    assert audio_span.attributes.get("gen_ai.usage.audio.input_tokens") == 5
    assert audio_span.attributes.get("gen_ai.usage.audio.output_tokens") == 10


def test_model_tool_definitions_gated_by_content_capture(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    monkeypatch,
) -> None:
    """Test that tool definitions are only prepared and recorded when content capture is enabled."""
    from opentelemetry.instrumentation.genai.agno import patch as agno_patch
    from opentelemetry.test_util_genai.instrumentor import instrument

    calls: list[Any] = []
    orig_prepare = agno_patch.prepare_tool_definitions

    def _spy_prepare(tools: Any) -> Any:
        calls.append(tools)
        return orig_prepare(tools)

    monkeypatch.setattr(agno_patch, "prepare_tool_definitions", _spy_prepare)

    tool_schema = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup item",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-tool-defs-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi back"},
                "finish_reason": "stop",
            }
        ],
    }
    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        ),
    )
    model = OpenAIChat(id="gpt-4o", client=client)

    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ):
        model.response(
            messages=[Message(role="user", content="hi")],
            tools=tool_schema,
        )

    assert calls == []
    (no_capture_span,) = span_exporter.get_finished_spans()
    assert (
        GenAIAttributes.GEN_AI_TOOL_DEFINITIONS
        not in no_capture_span.attributes
    )

    span_exporter.clear()
    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        model.response(
            messages=[Message(role="user", content="hi")],
            tools=tool_schema,
        )

    assert len(calls) == 1
    (capture_span,) = span_exporter.get_finished_spans()
    tool_defs: Any = capture_span.attributes[
        GenAIAttributes.GEN_AI_TOOL_DEFINITIONS
    ]
    assert tool_defs[0]["name"] == "lookup"


def test_model_stream_metadata_only_final_chunk(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that provider_data on a non-yielded metadata-only final chunk is recorded on both sync and async streams."""

    class _MetadataFinalStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="hello")
            yield ModelResponse(
                content=None,
                provider_data={
                    "id": "resp-meta-sync",
                    "model": "gpt-4o-meta-sync",
                    "finish_reason": "stop",
                },
                response_usage=MessageMetrics(
                    input_tokens=11, output_tokens=4
                ),
            )

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="async hello")
            yield ModelResponse(
                content=None,
                provider_data={
                    "id": "resp-meta-async",
                    "model": "claude-meta-async",
                    "stop_reason": "end_turn",
                },
                response_usage=MessageMetrics(
                    input_tokens=13, output_tokens=5
                ),
            )

    model = _MetadataFinalStreamModel(id="m", provider="OpenAI")
    chunks = list(
        model.response_stream(messages=[Message(role="user", content="hi")])
    )
    assert any(c.content == "hello" for c in chunks)

    (sync_span,) = span_exporter.get_finished_spans()
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-meta-sync"
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "gpt-4o-meta-sync"
    )
    assert sync_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("stop",)

    span_exporter.clear()

    async def _run_async() -> list[Any]:
        return [
            c
            async for c in model.aresponse_stream(
                messages=[Message(role="user", content="hi")]
            )
        ]

    async_chunks = asyncio.run(_run_async())
    assert any(c.content == "async hello" for c in async_chunks)

    (async_span,) = span_exporter.get_finished_spans()
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-meta-async"
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "claude-meta-async"
    )
    assert async_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("stop",)


def test_model_stream_early_close_and_error_preserves_media_output(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that closing or failing a sync or async stream after a media-only chunk records the buffered media output."""
    from agno.media import Audio, Image

    class _MediaStreamModel(MockModel):
        def __init__(self, *, fail: bool = False, **kwargs: Any) -> None:
            self.fail = fail
            super().__init__(**kwargs)

        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                images=[
                    Image(
                        url="https://example.com/stream-img.png",
                        mime_type="image/png",
                    )
                ],
                audio=Audio(content=b"\x01\x02", format="mp3"),
            )
            if self.fail:
                raise ConnectionError("Stream failed after media chunk")
            yield ModelResponse(content="later text")

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                images=[
                    Image(
                        url="https://example.com/async-stream-img.png",
                        mime_type="image/png",
                    )
                ],
            )
            if self.fail:
                raise ConnectionError("Async stream failed after media chunk")
            yield ModelResponse(content="later async text")

    # 1. Sync early close
    sync_model = _MediaStreamModel(id="m", provider="OpenAI")
    stream = sync_model.response_stream(
        messages=[Message(role="user", content="gen image")],
    )
    next(c for c in stream if c.images)
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    out_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert out_msgs[0]["parts"][0]["type"] == "uri"
    assert (
        out_msgs[0]["parts"][0]["uri"] == "https://example.com/stream-img.png"
    )
    assert out_msgs[0]["parts"][1]["type"] == "blob"
    assert out_msgs[0]["parts"][1]["modality"] == "audio"

    # 2. Sync mid-stream error
    span_exporter.clear()
    err_model = _MediaStreamModel(id="m", provider="OpenAI", fail=True)
    with pytest.raises(
        ConnectionError, match="Stream failed after media chunk"
    ):
        for _ in err_model.response_stream(
            messages=[Message(role="user", content="gen image")]
        ):
            pass

    (span,) = span_exporter.get_finished_spans()
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert (
        out_msgs[0]["parts"][0]["uri"] == "https://example.com/stream-img.png"
    )

    # 3. Async early close
    span_exporter.clear()

    async def _async_early_close() -> None:
        astream = sync_model.aresponse_stream(
            messages=[Message(role="user", content="gen image")],
        )
        chunk = await astream.__anext__()
        while not chunk.images:
            chunk = await astream.__anext__()
        await astream.aclose()

    asyncio.run(_async_early_close())
    (span,) = span_exporter.get_finished_spans()
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert (
        out_msgs[0]["parts"][0]["uri"]
        == "https://example.com/async-stream-img.png"
    )

    # 4. Async mid-stream error
    span_exporter.clear()

    async def _async_error() -> None:
        async for _ in err_model.aresponse_stream(
            messages=[Message(role="user", content="gen image")]
        ):
            pass

    with pytest.raises(
        ConnectionError, match="Async stream failed after media chunk"
    ):
        asyncio.run(_async_error())

    (span,) = span_exporter.get_finished_spans()
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert (
        out_msgs[0]["parts"][0]["uri"]
        == "https://example.com/async-stream-img.png"
    )


def test_anthropic_prompt_cache_tokens_normalized_in_input_tokens(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Anthropic-family models normalize input_tokens to include cache read and write tokens."""

    class _AnthropicCacheModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                content="Cached response",
                response_usage=MessageMetrics(
                    input_tokens=10,
                    output_tokens=25,
                    cache_read_tokens=100,
                    cache_write_tokens=20,
                ),
            )

        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="Cached stream",
                response_usage=MessageMetrics(
                    input_tokens=10,
                    output_tokens=25,
                    cache_read_tokens=100,
                    cache_write_tokens=20,
                ),
            )

    model = _AnthropicCacheModel(id="claude-sonnet-4-5", provider="Anthropic")
    model.response(messages=[Message(role="user", content="Hi")])

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 130
    )
    assert (
        span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
        )
        == 100
    )
    assert span.attributes.get("gen_ai.usage.cache_write.input_tokens") == 20
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 25
    )

    span_exporter.clear()
    list(model.response_stream(messages=[Message(role="user", content="Hi")]))
    (stream_span,) = span_exporter.get_finished_spans()
    assert (
        stream_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        == 130
    )


def test_model_tool_result_structured_response_null_and_media(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that tool result messages preserve structured AnyValue responses, None, and attached media."""
    from agno.media import Audio, File, Image, Video

    model = _SyncModel(id="gpt-4o", provider="OpenAI")
    messages = [
        Message(
            role="tool",
            tool_call_id="call_struct",
            content=[
                {"item": "apple", "count": 3},
                {"item": "pear", "count": 5},
            ],
            images=[
                Image(url="https://example.com/tool-img.png", format="png")
            ],
            audio=[Audio(content=b"\x00\x01", format="wav")],
            videos=[Video(filepath="/tmp/tool.mp4", format="mp4")],
            files=[File(url="https://example.com/tool.pdf")],
        ),
        Message(
            role="tool",
            tool_call_id="call_null",
            content=None,
        ),
    ]

    model.response(messages=messages)

    (span,) = span_exporter.get_finished_spans()
    input_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    assert len(input_msgs) == 2

    first_parts = input_msgs[0]["parts"]
    assert first_parts[0] == {
        "type": "tool_call_response",
        "id": "call_struct",
        "response": (
            {"item": "apple", "count": 3},
            {"item": "pear", "count": 5},
        ),
    }
    part_types = [p["type"] for p in first_parts[1:]]
    assert part_types == ["uri", "blob", "uri", "uri"]

    second_parts = list(input_msgs[1]["parts"])
    assert second_parts == [
        {
            "type": "tool_call_response",
            "id": "call_null",
            "response": None,
        }
    ]


def test_model_input_message_content_blocks(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that a list of provider content blocks on Message.content is parsed into typed message parts."""
    from base64 import b64encode

    raw_png = b"\x89PNG\r\n\x1a\n"
    raw_wav = b"RIFF\x00\x00\x00\x00WAVE"
    png_b64 = b64encode(raw_png).decode()
    wav_b64 = b64encode(raw_wav).decode()

    model = _SyncModel(id="gpt-4o", provider="OpenAI")
    messages = [
        Message(
            role="user",
            content=[
                {"type": "text", "text": "Describe this"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": png_b64,
                    },
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_b64, "format": "wav"},
                },
                {
                    "type": "file",
                    "file": {"file_id": "file-xyz", "filename": "report.pdf"},
                },
            ],
        )
    ]

    model.response(messages=messages)

    (span,) = span_exporter.get_finished_spans()
    input_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    parts = list(input_msgs[0]["parts"])
    assert parts == [
        {"type": "text", "content": "Describe this"},
        {
            "type": "uri",
            "modality": "image",
            "uri": "https://example.com/cat.png",
        },
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": raw_png,
        },
        {
            "type": "blob",
            "modality": "audio",
            "mime_type": "audio/wav",
            "content": raw_wav,
        },
        {
            "type": "file",
            "modality": "document",
            "mime_type": "application/pdf",
            "file_id": "file-xyz",
        },
    ]


def test_model_nested_anthropic_document_sources_and_roles(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test nested Anthropic document source blocks and custom/aliased message roles via model.response."""
    import base64

    model = _SyncModel(id="claude-3-7-sonnet", provider="Anthropic")
    pdf_b64 = base64.b64encode(b"%PDF-1.4 sample").decode("ascii")

    messages = [
        Message(
            role="developer",
            content="Follow developer policy.",
        ),
        Message(
            role="model",
            content="Understood.",
        ),
        Message(
            role="user",
            content=[
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "url",
                        "url": "https://example.com/spec.pdf",
                        "media_type": "application/pdf",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "text",
                        "media_type": "text/plain",
                        "data": "Plain text doc",
                    },
                },
                {
                    "type": "document",
                    "source": {
                        "type": "file",
                        "file_id": "file-doc-456",
                        "media_type": "application/pdf",
                    },
                },
            ],
        ),
    ]

    model.response(messages=messages)

    (span,) = span_exporter.get_finished_spans()
    input_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    assert [m["role"] for m in input_msgs] == [
        "developer",
        "assistant",
        "user",
    ]
    assert list(input_msgs[2]["parts"]) == [
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": b"%PDF-1.4 sample",
        },
        {
            "type": "uri",
            "modality": "document",
            "mime_type": "application/pdf",
            "uri": "https://example.com/spec.pdf",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "text/plain",
            "content": b"Plain text doc",
        },
        {
            "type": "file",
            "modality": "document",
            "mime_type": "application/pdf",
            "file_id": "file-doc-456",
        },
    ]


def test_real_openai_chat_sync_and_async(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test sync and async inference through the real OpenAIChat provider adapter with a transport mock."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-real-adapter-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response from OpenAIChat!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 18,
            "completion_tokens": 9,
            "total_tokens": 27,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    }

    sync_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )
        ),
    )
    model = OpenAIChat(id="gpt-4o", temperature=0.5, client=sync_client)
    resp = model.response(messages=[Message(role="user", content="Hello")])
    assert resp.content == "Response from OpenAIChat!"

    (sync_span,) = span_exporter.get_finished_spans()
    assert sync_span.name == "chat gpt-4o"
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "openai"
    )
    if resp.provider_data and "id" in resp.provider_data:
        assert (
            sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "chatcmpl-real-adapter-1"
        )
    else:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID not in sync_span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in sync_span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        not in sync_span.attributes
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        == 18
    )
    assert (
        sync_span.attributes.get(
            GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
        )
        == 4
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
        == 9
    )
    out_msgs: Any = sync_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    assert list(out_msgs[0]["parts"]) == [
        {"type": "text", "content": "Response from OpenAIChat!"}
    ]

    span_exporter.clear()

    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )
        ),
    )
    async_model = OpenAIChat(
        id="gpt-4o", temperature=0.5, async_client=async_client
    )
    async_resp = asyncio.run(
        async_model.aresponse(
            messages=[Message(role="user", content="Async hello")]
        )
    )
    assert async_resp.content == "Response from OpenAIChat!"

    (async_span,) = span_exporter.get_finished_spans()
    assert async_span.name == "chat gpt-4o"
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "openai"
    )
    if async_resp.provider_data and "id" in async_resp.provider_data:
        assert (
            async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "chatcmpl-real-adapter-1"
        )
    else:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID not in async_span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in async_span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        not in async_span.attributes
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        == 18
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
        == 9
    )


def test_real_openai_chat_stream_sync_and_async(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test sync and async streaming inference through the real OpenAIChat adapter over SSE."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    sse_lines = [
        'data: {"id":"chatcmpl-sse-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello "},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-sse-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[{"index":0,"delta":{"content":"world!"},"finish_reason":"stop"}]}',
        'data: {"id":"chatcmpl-sse-1","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-2024-08-06","choices":[],"usage":{"prompt_tokens":11,"completion_tokens":4,"total_tokens":15}}',
        "data: [DONE]",
    ]
    sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")

    sync_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body,
                )
            )
        ),
    )
    model = OpenAIChat(id="gpt-4o", client=sync_client)
    chunks = list(
        model.response_stream(
            messages=[Message(role="user", content="Stream")]
        )
    )
    assert "".join(c.content or "" for c in chunks) == "Hello world!"

    (sync_span,) = span_exporter.get_finished_spans()
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
    )
    has_stream_id = any(
        c.provider_data and "id" in c.provider_data for c in chunks
    )
    if has_stream_id:
        assert (
            sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "chatcmpl-sse-1"
        )
    else:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID not in sync_span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in sync_span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        not in sync_span.attributes
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        == 11
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
        == 4
    )
    out_msgs: Any = sync_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    assert list(out_msgs[0]["parts"]) == [
        {"type": "text", "content": "Hello world!"}
    ]

    span_exporter.clear()

    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body,
                )
            )
        ),
    )
    async_model = OpenAIChat(id="gpt-4o", async_client=async_client)

    async def _consume_async() -> str:
        parts: list[str] = []
        async for chunk in async_model.aresponse_stream(
            messages=[Message(role="user", content="Async stream")]
        ):
            if chunk.content:
                parts.append(chunk.content)
        return "".join(parts)

    assert asyncio.run(_consume_async()) == "Hello world!"

    (async_span,) = span_exporter.get_finished_spans()
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM)
        is True
    )
    if has_stream_id:
        assert (
            async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "chatcmpl-sse-1"
        )
    else:
        assert GenAIAttributes.GEN_AI_RESPONSE_ID not in async_span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in async_span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        not in async_span.attributes
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS)
        == 11
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS)
        == 4
    )


def test_openai_stream_partial_tool_call_deltas_early_close_and_error(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that early close or mid-stream error on an OpenAI tool-call stream normalizes ChoiceDeltaToolCall deltas without raising AttributeError."""
    from typing import cast

    from agno.models.openai import OpenAIChat
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    class _DeltaToolStreamModel(OpenAIChat):
        def __init__(self, *, fail: bool = False, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._fail = fail

        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                tool_calls=cast(
                    Any,
                    [
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_delta_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name="get_weather",
                                arguments='{"city": ',
                            ),
                        )
                    ],
                )
            )
            yield ModelResponse(
                tool_calls=cast(
                    Any,
                    [
                        ChoiceDeltaToolCall(
                            index=0,
                            id=None,
                            type=None,
                            function=ChoiceDeltaToolCallFunction(
                                name=None,
                                arguments='"Paris"}',
                            ),
                        )
                    ],
                )
            )
            if self._fail:
                raise ConnectionError("Tool stream interrupted")
            yield ModelResponse(content="unreachable")

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                tool_calls=cast(
                    Any,
                    [
                        ChoiceDeltaToolCall(
                            index=0,
                            id="call_delta_async_1",
                            type="function",
                            function=ChoiceDeltaToolCallFunction(
                                name="get_weather",
                                arguments='{"city": "Tokyo"}',
                            ),
                        )
                    ],
                )
            )
            if self._fail:
                raise ConnectionError("Async tool stream interrupted")
            yield ModelResponse(content="unreachable")

    # 1. Sync early close after partial tool-call deltas
    model = _DeltaToolStreamModel(id="gpt-4o", api_key="test")
    stream = model.response_stream(
        messages=[Message(role="user", content="Weather?")],
    )
    seen_tool_deltas = 0
    for chunk in stream:
        if chunk.tool_calls:
            seen_tool_deltas += 1
            if seen_tool_deltas == 2:
                break
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    out_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert list(out_msgs[0]["parts"]) == [
        {
            "type": "tool_call",
            "id": "call_delta_1",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
    ]

    # 2. Sync mid-stream error after partial tool-call deltas
    span_exporter.clear()
    err_model = _DeltaToolStreamModel(id="gpt-4o", api_key="test", fail=True)
    with pytest.raises(ConnectionError, match="Tool stream interrupted"):
        for _ in err_model.response_stream(
            messages=[Message(role="user", content="Weather?")]
        ):
            pass

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ConnectionError"
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert list(out_msgs[0]["parts"]) == [
        {
            "type": "tool_call",
            "id": "call_delta_1",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        }
    ]

    # 3. Async early close after partial tool-call deltas
    span_exporter.clear()

    async def _async_close() -> None:
        astream = model.aresponse_stream(
            messages=[Message(role="user", content="Weather?")],
        )
        chunk = await astream.__anext__()
        while not chunk.tool_calls:
            chunk = await astream.__anext__()
        await astream.aclose()

    asyncio.run(_async_close())
    (span,) = span_exporter.get_finished_spans()
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert list(out_msgs[0]["parts"]) == [
        {
            "type": "tool_call",
            "id": "call_delta_async_1",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
        }
    ]

    # 4. Async mid-stream error after partial tool-call deltas
    span_exporter.clear()

    async def _async_err() -> None:
        async for _ in err_model.aresponse_stream(
            messages=[Message(role="user", content="Weather?")]
        ):
            pass

    with pytest.raises(ConnectionError, match="Async tool stream interrupted"):
        asyncio.run(_async_err())

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "ConnectionError"
    out_msgs = span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert list(out_msgs[0]["parts"]) == [
        {
            "type": "tool_call",
            "id": "call_delta_async_1",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
        }
    ]


def test_real_openai_request_params_override_and_zero_max_tokens(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that request_params overrides model fields and preserves zero max_tokens."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    sent_bodies: list[dict[str, Any]] = []

    def _transport(request: httpx.Request) -> httpx.Response:
        sent_bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-override-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-2024-08-06",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_transport)),
    )
    model = OpenAIChat(
        id="gpt-4o",
        temperature=0.7,
        top_p=0.9,
        max_tokens=0,
        max_completion_tokens=100,
        request_params={
            "temperature": 0,
            "top_p": None,
            "frequency_penalty": 0.25,
            "stop": ["STOP_HERE"],
        },
        client=client,
    )
    model.response(messages=[Message(role="user", content="Hi")])

    assert sent_bodies[0]["temperature"] == 0
    assert sent_bodies[0]["top_p"] is None
    assert sent_bodies[0]["max_tokens"] == 0
    assert sent_bodies[0]["frequency_penalty"] == 0.25
    assert sent_bodies[0]["stop"] == ["STOP_HERE"]

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.0
    )
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_P not in span.attributes
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 0
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY)
        == 0.25
    )
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES
    ) == ("STOP_HERE",)

    span_exporter.clear()
    null_temp_model = OpenAIChat(
        id="gpt-4o",
        temperature=0.7,
        request_params={"temperature": None},
        client=client,
    )
    null_temp_model.response(messages=[Message(role="user", content="Hi 2")])

    assert sent_bodies[1]["temperature"] is None
    (null_temp_span,) = span_exporter.get_finished_spans()
    assert (
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE
        not in null_temp_span.attributes
    )


def test_model_stream_defers_invocation_until_first_read(
    instrument_agno,
    span_exporter,
    tracer_provider,
) -> None:
    """Test that sync and async stream wrappers start inference on first read under the consumption context."""
    tracer = tracer_provider.get_tracer(__name__)
    model = _StreamModel(id="lazy-stream-model", provider="OpenAI")

    with tracer.start_as_current_span("construct_parent"):
        stream = model.response_stream(
            messages=[Message(role="user", content="Lazy sync")]
        )
        assert len(span_exporter.get_finished_spans()) == 0

    with tracer.start_as_current_span("consume_parent") as consume_span:
        list(stream)

    spans = span_exporter.get_finished_spans()
    chat_span = next(s for s in spans if s.name == "chat lazy-stream-model")
    assert chat_span.parent is not None
    assert chat_span.parent.span_id == consume_span.get_span_context().span_id

    span_exporter.clear()

    async def _run_async() -> int:
        with tracer.start_as_current_span("async_construct_parent"):
            astream = model.aresponse_stream(
                messages=[Message(role="user", content="Lazy async")]
            )
            assert len(span_exporter.get_finished_spans()) == 0

        with tracer.start_as_current_span(
            "async_consume_parent"
        ) as aconsume_span:
            async for _ in astream:
                pass
            return aconsume_span.get_span_context().span_id

    expected_parent_id = asyncio.run(_run_async())
    aspans = span_exporter.get_finished_spans()
    achat_span = next(s for s in aspans if s.name == "chat lazy-stream-model")
    assert achat_span.parent is not None
    assert achat_span.parent.span_id == expected_parent_id


def test_model_stream_preserves_multi_image_multi_chunk_output(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that sync and async stream wrappers preserve all images within a chunk and across chunks."""
    from agno.media import Image

    class _MultiImageStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="part 1 ",
                images=[
                    Image(
                        url="https://example.com/img1.png",
                        mime_type="image/png",
                    ),
                    Image(
                        url="https://example.com/img2.png",
                        mime_type="image/png",
                    ),
                ],
            )
            yield ModelResponse(
                content="part 2",
                images=[
                    Image(
                        url="https://example.com/img3.png",
                        mime_type="image/png",
                    ),
                ],
            )

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(
                content="apart 1 ",
                images=[
                    Image(
                        url="https://example.com/aimg1.png",
                        mime_type="image/png",
                    ),
                    Image(
                        url="https://example.com/aimg2.png",
                        mime_type="image/png",
                    ),
                ],
            )
            yield ModelResponse(
                content="apart 2",
                images=[
                    Image(
                        url="https://example.com/aimg3.png",
                        mime_type="image/png",
                    ),
                ],
            )

    model = _MultiImageStreamModel(id="multi-img", provider="OpenAI")
    list(
        model.response_stream(messages=[Message(role="user", content="imgs")])
    )

    (sync_span,) = span_exporter.get_finished_spans()
    sync_out: Any = sync_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    sync_uris = [
        p["uri"] for p in sync_out[0]["parts"] if p.get("type") == "uri"
    ]
    assert sync_uris == [
        "https://example.com/img1.png",
        "https://example.com/img2.png",
        "https://example.com/img3.png",
    ]

    span_exporter.clear()

    async def _consume() -> None:
        async for _ in model.aresponse_stream(
            messages=[Message(role="user", content="aimgs")]
        ):
            pass

    asyncio.run(_consume())
    (async_span,) = span_exporter.get_finished_spans()
    async_out: Any = async_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    async_uris = [
        p["uri"] for p in async_out[0]["parts"] if p.get("type") == "uri"
    ]
    assert async_uris == [
        "https://example.com/aimg1.png",
        "https://example.com/aimg2.png",
        "https://example.com/aimg3.png",
    ]


def test_real_openai_chat_audio_response_and_stream_not_double_encoded(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that base64 audio from OpenAIChat (normal and streamed) is decoded once into BlobPart."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-audio-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-audio-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "audio": {
                        "id": "audio_123",
                        "data": "AAE=",
                        "expires_at": 1700003600,
                        "transcript": "Hello audio",
                    },
                },
                "finish_reason": "stop",
            }
        ],
    }
    sync_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        ),
    )
    model = OpenAIChat(id="gpt-4o-audio-preview", client=sync_client)
    model.response(messages=[Message(role="user", content="Speak")])

    (sync_span,) = span_exporter.get_finished_spans()
    sync_out: Any = sync_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    audio_blobs = [
        p
        for p in sync_out[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(audio_blobs) == 1
    assert audio_blobs[0]["content"] == b"\x00\x01"

    span_exporter.clear()

    sse_lines = [
        'data: {"id":"chatcmpl-audio-s","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-audio-preview","choices":[{"index":0,"delta":{"role":"assistant","audio":{"id":"audio_s1","data":"AA","transcript":"Hello "}},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-audio-s","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o-audio-preview","choices":[{"index":0,"delta":{"audio":{"data":"E=","transcript":"audio"}},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")
    stream_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body,
                )
            )
        ),
    )
    stream_model = OpenAIChat(id="gpt-4o-audio-preview", client=stream_client)
    list(
        stream_model.response_stream(
            messages=[Message(role="user", content="Speak stream")]
        )
    )

    (stream_span,) = span_exporter.get_finished_spans()
    stream_out: Any = stream_span.attributes[
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    ]
    stream_audio_blobs = [
        p
        for p in stream_out[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(stream_audio_blobs) == 1
    assert stream_audio_blobs[0]["content"] == b"\x00\x01"

    span_exporter.clear()

    from agno.media import Audio

    raw_b64_looking_audio = b"AAAA"

    class _RawAudioModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                content="Raw PCM audio",
                audio=Audio(content=raw_b64_looking_audio, format="wav"),
            )

    raw_model = _RawAudioModel(id="gemini-2.5-flash", provider="Google")
    raw_model.response(
        messages=[
            Message(
                role="user",
                content="Check raw audio",
                audio=[Audio(content=raw_b64_looking_audio, format="wav")],
            )
        ]
    )

    (raw_span,) = span_exporter.get_finished_spans()
    raw_in: Any = raw_span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    raw_in_blobs = [
        p
        for p in raw_in[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(raw_in_blobs) == 1
    assert raw_in_blobs[0]["content"] == raw_b64_looking_audio

    raw_out: Any = raw_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    raw_out_blobs = [
        p
        for p in raw_out[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(raw_out_blobs) == 1
    assert raw_out_blobs[0]["content"] == raw_b64_looking_audio


def test_model_video_url_content_blocks(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test that provider-native video_url content blocks in Message.content are captured as video UriPart and BlobPart."""
    model = _SyncModel(id="kimi-latest", provider="MoonShot")
    messages = [
        Message(
            role="user",
            content=[
                {
                    "type": "video_url",
                    "video_url": {"url": "ms://vid-ref-123"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "data:video/mp4;base64,AAE="},
                },
            ],
        )
    ]

    model.response(messages=messages)

    (span,) = span_exporter.get_finished_spans()
    input_msgs: Any = span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    assert len(input_msgs) == 1
    parts = input_msgs[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["type"] == "uri"
    assert parts[0]["modality"] == "video"
    assert parts[0]["uri"] == "ms://vid-ref-123"
    assert parts[1] == {
        "type": "blob",
        "modality": "video",
        "mime_type": "video/mp4",
        "content": b"\x00\x01",
    }


def test_model_stream_throw_and_athrow_finalize_inner_wrapper(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that stream.throw() and stream.athrow() on response_stream/aresponse_stream finalize and close the inner stream."""
    import gc

    sync_closed = False
    async_closed = False

    class _ThrowStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal sync_closed
            try:
                yield ModelResponse(
                    content="chunk 1",
                    provider_data={
                        "id": "resp-throw-1",
                        "finish_reason": "length",
                    },
                )
                yield ModelResponse(content="chunk 2")
            finally:
                sync_closed = True

        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal async_closed
            _retained = self.retained_caller_stream
            assert _retained is not None or _retained is None
            try:
                yield ModelResponse(
                    content="async chunk 1",
                    provider_data={
                        "id": "resp-athrow-1",
                        "stop_reason": "max_tokens",
                    },
                )
                yield ModelResponse(content="async chunk 2")
            finally:
                async_closed = True

    model = _ThrowStreamModel(id="throw-model", provider="OpenAI")
    model.retained_caller_stream = None
    stream = model.response_stream(
        messages=[Message(role="user", content="Hi")],
    )
    first = next(c for c in stream if c.content is not None)
    assert first.content == "chunk 1"

    with pytest.raises(ValueError, match="Injected sync error"):
        stream.throw(ValueError("Injected sync error"))
    assert sync_closed
    assert stream._self_inner_stream._self_stream.gi_frame is None
    del stream
    gc.collect()

    (sync_span,) = span_exporter.get_finished_spans()
    assert sync_span.status.status_code == StatusCode.ERROR
    assert sync_span.attributes.get(ERROR_TYPE) == "ValueError"
    assert sync_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("length",)

    span_exporter.clear()

    caller_stream_closed = False

    async def _caller_owned_stream() -> Any:
        nonlocal caller_stream_closed
        try:
            yield "caller-item-1"
            yield "caller-item-2"
        finally:
            caller_stream_closed = True

    async def _run_async() -> None:
        caller_gen = _caller_owned_stream()
        assert await caller_gen.__anext__() == "caller-item-1"
        model.retained_caller_stream = caller_gen

        astream = model.aresponse_stream(
            messages=[Message(role="user", content="Hi async")],
        )
        afirst = await astream.__anext__()
        while afirst.content is None:
            afirst = await astream.__anext__()
        assert afirst.content == "async chunk 1"

        with pytest.raises(ValueError, match="Injected async error"):
            await astream.athrow(ValueError("Injected async error"))
        assert async_closed
        assert not caller_stream_closed
        assert await caller_gen.__anext__() == "caller-item-2"
        await caller_gen.aclose()
        assert caller_stream_closed
        assert astream._self_inner_stream._self_stream.ag_frame is None
        del astream
        gc.collect()

    asyncio.run(_run_async())

    (async_span,) = span_exporter.get_finished_spans()
    assert async_span.status.status_code == StatusCode.ERROR
    assert async_span.attributes.get(ERROR_TYPE) == "ValueError"
    assert async_span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("length",)


def test_model_injected_client_endpoint_overrides_adapter_base_url(
    instrument_agno,
    span_exporter,
    metric_reader,
) -> None:
    """Test that injected client/async_client endpoints override adapter base_url on spans and metrics without eagerly constructing optional clients."""
    import httpx
    import openai
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-endpoint-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "total_tokens": 12,
        },
    }

    sync_client = openai.OpenAI(
        api_key="test-key",
        base_url="https://sync-injected.example.com:8443/v1",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        ),
    )
    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://async-injected.example.com:9443/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        ),
    )

    model = OpenAIChat(
        id="gpt-4o",
        base_url="https://adapter-unused.example.com:7000/v1",
        client=sync_client,
        async_client=async_client,
    )
    model.response(messages=[Message(role="user", content="sync check")])
    asyncio.run(
        model.aresponse(messages=[Message(role="user", content="async check")])
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sync_span, async_span = spans
    assert sync_span.attributes.get("server.address") == (
        "sync-injected.example.com"
    )
    assert sync_span.attributes.get("server.port") == 8443
    assert async_span.attributes.get("server.address") == (
        "async-injected.example.com"
    )
    assert async_span.attributes.get("server.port") == 9443

    metrics_data = metric_reader.get_metrics_data()
    duration_endpoints: set[tuple[Any, Any]] = set()
    token_endpoints: set[tuple[Any, Any]] = set()
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        duration_endpoints.add(
                            (
                                dp.attributes.get("server.address"),
                                dp.attributes.get("server.port"),
                            )
                        )
                elif metric.name == "gen_ai.client.token.usage":
                    for dp in metric.data.data_points:
                        token_endpoints.add(
                            (
                                dp.attributes.get("server.address"),
                                dp.attributes.get("server.port"),
                            )
                        )
    expected_endpoints = {
        ("sync-injected.example.com", 8443),
        ("async-injected.example.com", 9443),
    }
    assert expected_endpoints.issubset(duration_endpoints)
    assert expected_endpoints.issubset(token_endpoints)

    # Also verify an adapter without injected client resolves its own base_url
    # without eagerly constructing optional clients before invoke().
    span_exporter.clear()
    constructed_During_start: list[bool] = []

    class _LazyClientModel(MockModel):
        client: Any = None
        async_client: Any = None
        base_url: str = "https://lazy-adapter.example.com:6443/v1"

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            constructed_During_start.append(
                self.client is not None or self.async_client is not None
            )
            return ModelResponse(content="lazy ok")

    lazy_model = _LazyClientModel(id="lazy-m", provider="OpenAI")
    lazy_model.response(messages=[Message(role="user", content="lazy")])
    assert constructed_During_start == [False]
    (lazy_span,) = span_exporter.get_finished_spans()
    assert (
        lazy_span.attributes.get("server.address")
        == "lazy-adapter.example.com"
    )
    assert lazy_span.attributes.get("server.port") == 6443


def test_model_async_stream_copy_safe_and_uninstrument_restores_delegates(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that async streaming hooks are copy-safe on cloned models and are restored by uninstrument()."""
    import copy

    import httpx
    import openai
    from agno.models.base import Model
    from agno.models.openai import OpenAIChat

    sse_lines = [
        'data: {"id":"chatcmpl-copy","object":"chat.completion.chunk","created":1700000000,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")

    orig_requests: list[str] = []
    copy_requests: list[str] = []

    def _orig_transport(req: httpx.Request) -> httpx.Response:
        orig_requests.append(json.loads(req.content.decode("utf-8"))["model"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    def _copy_transport(req: httpx.Request) -> httpx.Response:
        copy_requests.append(json.loads(req.content.decode("utf-8"))["model"])
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    orig_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://orig-stream.example.com:8443/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_orig_transport)
        ),
    )
    copy_client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://copy-stream.example.com:9443/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_copy_transport)
        ),
    )

    orig_model = OpenAIChat(id="gpt-4o-orig", async_client=orig_client)

    async def _consume(m: OpenAIChat) -> None:
        async for _ in m.aresponse_stream(
            messages=[Message(role="user", content="stream")]
        ):
            pass

    asyncio.run(_consume(orig_model))
    assert "ainvoke_stream" not in orig_model.__dict__
    assert "_ainvoke_stream_with_retry" not in orig_model.__dict__

    copied_model = copy.copy(orig_model)
    copied_model.id = "gpt-4o-copy"
    copied_model.async_client = copy_client

    asyncio.run(_consume(copied_model))

    assert orig_requests == ["gpt-4o-orig"]
    assert copy_requests == ["gpt-4o-copy"]

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert (
        spans[0].attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "gpt-4o-orig"
    )
    assert spans[0].attributes.get("server.address") == (
        "orig-stream.example.com"
    )
    assert (
        spans[1].attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
        == "gpt-4o-copy"
    )
    assert spans[1].attributes.get("server.address") == (
        "copy-stream.example.com"
    )

    # Uninstrument and verify delegate methods are restored on the class
    AgnoInstrumentor().uninstrument()
    assert not hasattr(OpenAIChat.__dict__["ainvoke_stream"], "__wrapped__")
    if hasattr(Model, "_ainvoke_stream_with_retry"):
        assert not hasattr(
            Model.__dict__["_ainvoke_stream_with_retry"], "__wrapped__"
        )


def test_model_async_stream_delegating_subclass_before_base_class(
    instrument_agno,
    span_exporter,
    tracer_provider,
    logger_provider,
    meter_provider,
) -> None:
    """Streaming a delegating subclass (e.g. MiniMax) before its base class wraps all hierarchy levels and awaits base generator cleanup on early aclose/athrow."""
    import httpx
    import openai
    from agno.models.base import Model
    from agno.models.openai import OpenAIChat

    try:
        from agno.models.minimax import MiniMax as _DelegatingOpenAISubclass
    except ImportError:

        class _DelegatingOpenAISubclass(OpenAIChat):
            async def ainvoke_stream(
                self, *args: Any, **kwargs: Any
            ) -> AsyncIterator[ModelResponse]:
                async for response in super().ainvoke_stream(*args, **kwargs):
                    yield response

    # Re-instrument cleanly so OpenAIChat.ainvoke_stream is not pre-wrapped by earlier tests.
    AgnoInstrumentor().uninstrument()
    AgnoInstrumentor().instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    assert not hasattr(OpenAIChat.__dict__["ainvoke_stream"], "__wrapped__")
    assert not hasattr(
        _DelegatingOpenAISubclass.__dict__["ainvoke_stream"], "__wrapped__"
    )

    sse_lines = [
        'data: {"id":"chatcmpl-mm-1","object":"chat.completion.chunk","created":1700000000,"model":"MiniMax-M2.1","choices":[{"index":0,"delta":{"role":"assistant","content":"chunk1"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-mm-1","object":"chat.completion.chunk","created":1700000000,"model":"MiniMax-M2.1","choices":[{"index":0,"delta":{"content":"chunk2"},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    sse_body = ("\n\n".join(sse_lines) + "\n\n").encode("utf-8")

    client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.minimax.io/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=sse_body,
                )
            )
        ),
    )
    minimax_model = _DelegatingOpenAISubclass(
        id="MiniMax-M2.1", api_key="test-key", async_client=client
    )

    # Also test a delegating hierarchy with explicit async cleanup in the base generator
    base_cleanups: list[str] = []

    class _BaseAsyncStreamModel(Model):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            yield ModelResponse(content="unused")

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                yield ModelResponse(content="first")
                yield ModelResponse(content="second")
            finally:
                await asyncio.sleep(0)
                base_cleanups.append("closed")

        def _parse_provider_response(
            self, response: Any, **kwargs: Any
        ) -> ModelResponse:
            return response

        def _parse_provider_response_delta(
            self, response: Any
        ) -> ModelResponse:
            return response

    class _DelegatingSubStreamModel(_BaseAsyncStreamModel):
        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            async for delta in super().ainvoke_stream(*args, **kwargs):
                yield delta

    sub_model = _DelegatingSubStreamModel(id="sub-model", provider="Custom")

    async def _next_content_chunk(stream: Any) -> Any:
        chunk = await anext(stream)
        while getattr(chunk, "content", None) is None:
            chunk = await anext(stream)
        return chunk

    async def _run() -> None:
        # 1. Stream MiniMax first (before OpenAIChat is ever streamed) and close early
        m_stream = minimax_model.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first_chunk = await _next_content_chunk(m_stream)
        assert getattr(first_chunk, "content", None) == "chunk1"
        assert hasattr(
            _DelegatingOpenAISubclass.__dict__["ainvoke_stream"], "__wrapped__"
        )
        assert hasattr(OpenAIChat.__dict__["ainvoke_stream"], "__wrapped__")
        await m_stream.aclose()

        # 2. Early aclose() on delegating subclass awaits base generator cleanup
        s_stream = sub_model.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        chunk = await _next_content_chunk(s_stream)
        assert getattr(chunk, "content", None) == "first"
        assert hasattr(
            _DelegatingSubStreamModel.__dict__["ainvoke_stream"], "__wrapped__"
        )
        assert hasattr(
            _BaseAsyncStreamModel.__dict__["ainvoke_stream"], "__wrapped__"
        )
        await s_stream.aclose()
        assert base_cleanups == ["closed"]

        # 3. Early athrow() on delegating subclass also awaits base generator cleanup
        s_stream2 = sub_model.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first2 = await _next_content_chunk(s_stream2)
        assert getattr(first2, "content", None) == "first"
        with pytest.raises(RuntimeError, match="caller abort"):
            await s_stream2.athrow(RuntimeError("caller abort"))
        assert base_cleanups == ["closed", "closed"]

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    AgnoInstrumentor().uninstrument()
    assert not hasattr(
        _DelegatingOpenAISubclass.__dict__["ainvoke_stream"], "__wrapped__"
    )
    assert not hasattr(OpenAIChat.__dict__["ainvoke_stream"], "__wrapped__")
    assert not hasattr(
        _DelegatingSubStreamModel.__dict__["ainvoke_stream"], "__wrapped__"
    )
    assert not hasattr(
        _BaseAsyncStreamModel.__dict__["ainvoke_stream"], "__wrapped__"
    )


def test_model_async_stream_delegate_close_error_still_closes_remaining_delegates(
    instrument_agno,
    span_exporter,
) -> None:
    """If a delegate generator's aclose() raises, remaining delegates are still closed and the first close error is propagated."""

    from agno.models.base import Model

    events: list[str] = []

    class _BaseFailingCloseStreamModel(Model):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            yield ModelResponse(content="unused")

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                yield ModelResponse(content="chunk1")
                yield ModelResponse(content="chunk2")
            finally:
                await asyncio.sleep(0)
                events.append("base_closing")
                raise RuntimeError("inner delegate close failed")

        def _parse_provider_response(
            self, response: Any, **kwargs: Any
        ) -> ModelResponse:
            return response

        def _parse_provider_response_delta(
            self, response: Any
        ) -> ModelResponse:
            return response

    class _SubFailingCloseStreamModel(_BaseFailingCloseStreamModel):
        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                async for delta in super().ainvoke_stream(*args, **kwargs):
                    yield delta
            finally:
                await asyncio.sleep(0)
                events.append("sub_closing")
                raise ValueError("middle delegate close failed")

    class _TopTrackedStreamModel(_SubFailingCloseStreamModel):
        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                async for delta in super().ainvoke_stream(*args, **kwargs):
                    yield delta
            finally:
                await asyncio.sleep(0)
                events.append("top_closed")

    async def _next_content_chunk(stream: Any) -> Any:
        chunk = await anext(stream)
        while getattr(chunk, "content", None) is None:
            chunk = await anext(stream)
        return chunk

    async def _run() -> None:
        # 1. Earlier delegate closer raises: remaining outer delegates still close and first delegate error propagates
        model = _TopTrackedStreamModel(
            id="top-tracked-close", provider="Custom"
        )
        stream = model.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first = await _next_content_chunk(stream)
        assert getattr(first, "content", None) == "chunk1"
        with pytest.raises(RuntimeError, match="inner delegate close failed"):
            await stream.aclose()
        assert events == ["base_closing", "sub_closing", "top_closed"]

        # 2. Wrapped stream close raises first: all delegates still close and wrapped stream error is preserved
        events.clear()
        model2 = _TopTrackedStreamModel(
            id="wrapped-fail-close", provider="Custom"
        )
        stream2 = model2.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first2 = await _next_content_chunk(stream2)
        assert getattr(first2, "content", None) == "chunk1"

        inner_wrapper = stream2._self_inner_stream
        orig_inner_stream = inner_wrapper._self_stream

        class _FailingWrappedStream:
            def __aiter__(self) -> Any:
                return self

            async def __anext__(self) -> Any:
                return await orig_inner_stream.__anext__()

            async def aclose(self) -> None:
                events.append("wrapped_closing")
                try:
                    await orig_inner_stream.aclose()
                finally:
                    raise ConnectionError("wrapped stream close failed")

        inner_wrapper._self_stream = _FailingWrappedStream()

        with pytest.raises(
            ConnectionError, match="wrapped stream close failed"
        ):
            await stream2.aclose()
        assert events == [
            "wrapped_closing",
            "base_closing",
            "sub_closing",
            "top_closed",
        ]

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes.get(ERROR_TYPE) == "RuntimeError"
    assert spans[1].status.status_code == StatusCode.ERROR
    assert spans[1].attributes.get(ERROR_TYPE) == "ConnectionError"


def test_model_async_stream_cancelled_inner_cleanup_preserves_original_error(
    instrument_agno,
    span_exporter,
) -> None:
    """CancelledError raised during inner-stream cleanup after an iteration or athrow error preserves the original error."""
    from agno.models.base import Model

    events: list[str] = []

    class _BaseCancelledCleanupStreamModel(Model):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            yield ModelResponse(content="unused")

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                yield ModelResponse(content="chunk1")
                yield ModelResponse(content="chunk2")
            finally:
                await asyncio.sleep(0)
                events.append("cleanup_cancelled")
                raise asyncio.CancelledError("cleanup cancelled")

        def _parse_provider_response(
            self, response: Any, **kwargs: Any
        ) -> ModelResponse:
            return response

        def _parse_provider_response_delta(
            self, response: Any
        ) -> ModelResponse:
            return response

    class _SubFailingMidStreamModel(_BaseCancelledCleanupStreamModel):
        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            async for delta in super().ainvoke_stream(*args, **kwargs):
                yield delta
                raise ConnectionError("stream failed mid-iteration")

    async def _next_content_chunk(stream: Any) -> Any:
        chunk = await anext(stream)
        while getattr(chunk, "content", None) is None:
            chunk = await anext(stream)
        return chunk

    async def _run() -> None:
        # 1. Mid-iteration provider error while base delegate is still open;
        #    inner-stream cleanup raises CancelledError, original ConnectionError must survive.
        model_mid = _SubFailingMidStreamModel(
            id="cancel-cleanup-mid", provider="Custom"
        )
        stream_mid = model_mid.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first = await _next_content_chunk(stream_mid)
        assert getattr(first, "content", None) == "chunk1"
        with pytest.raises(
            ConnectionError, match="stream failed mid-iteration"
        ):
            await anext(stream_mid)
        assert events == ["cleanup_cancelled"]

        # 2. Caller athrow() failure with CancelledError during inner cleanup
        events.clear()
        model_throw = _BaseCancelledCleanupStreamModel(
            id="cancel-cleanup-throw", provider="Custom"
        )
        stream_throw = model_throw.aresponse_stream(
            messages=[Message(role="user", content="hi")]
        )
        first2 = await _next_content_chunk(stream_throw)
        assert getattr(first2, "content", None) == "chunk1"
        with pytest.raises(RuntimeError, match="caller abort"):
            await stream_throw.athrow(RuntimeError("caller abort"))
        assert events == ["cleanup_cancelled"]

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes.get(ERROR_TYPE) == "ConnectionError"
    assert spans[1].status.status_code == StatusCode.ERROR
    assert spans[1].attributes.get(ERROR_TYPE) == "RuntimeError"


def test_openai_responses_preserves_response_id_sync_async_and_streaming(
    instrument_agno,
    span_exporter,
) -> None:
    """OpenAIResponses stores response_id in provider_data['response_id']; sync, async, and streaming preserve gen_ai.response.id."""
    import httpx
    import openai
    from agno.models.openai.responses import OpenAIResponses

    def _response_payload(resp_id: str, text: str) -> dict[str, Any]:
        return {
            "id": resp_id,
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "gpt-4o",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        }
                    ],
                }
            ],
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "usage": {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 15,
            },
        }

    def _sse_body(resp_id: str, text: str) -> bytes:
        completed = _response_payload(resp_id, text)
        created = {
            **completed,
            "status": "in_progress",
            "output": [],
            "usage": None,
        }
        events_sse = [
            f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': created})}",
            f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'sequence_number': 1, 'item_id': 'msg_1', 'output_index': 0, 'content_index': 0, 'delta': text, 'logprobs': []})}",
            f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'sequence_number': 2, 'response': completed})}",
        ]
        return ("\n\n".join(events_sse) + "\n\n").encode("utf-8")

    sync_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda req: (
                    httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=_sse_body(
                            "resp_stream_sync_1", "hello sync stream"
                        ),
                    )
                    if json.loads(req.content.decode("utf-8")).get("stream")
                    else httpx.Response(
                        200,
                        json=_response_payload("resp_sync_1", "hello sync"),
                    )
                )
            )
        ),
    )
    async_client = openai.AsyncOpenAI(
        api_key="test-key",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: (
                    httpx.Response(
                        200,
                        headers={"content-type": "text/event-stream"},
                        content=_sse_body(
                            "resp_stream_async_1", "hello async stream"
                        ),
                    )
                    if json.loads(req.content.decode("utf-8")).get("stream")
                    else httpx.Response(
                        200,
                        json=_response_payload("resp_async_1", "hello async"),
                    )
                )
            )
        ),
    )

    model = OpenAIResponses(
        id="gpt-4o", client=sync_client, async_client=async_client
    )

    # 1. Sync non-streaming
    model.response(messages=[Message(role="user", content="hi sync")])
    # 2. Sync streaming
    list(
        model.response_stream(
            messages=[Message(role="user", content="hi sync stream")]
        )
    )

    # 3. Async non-streaming + 4. Async streaming
    async def _run_async() -> None:
        await model.aresponse(
            messages=[Message(role="user", content="hi async")]
        )
        async for _ in model.aresponse_stream(
            messages=[Message(role="user", content="hi async stream")]
        ):
            pass

    asyncio.run(_run_async())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4
    assert [
        span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        for span in spans
    ] == [
        "resp_sync_1",
        "resp_stream_sync_1",
        "resp_async_1",
        "resp_stream_async_1",
    ]


def test_openai_responses_tool_call_correlates_via_call_id(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """OpenAIResponses tool loops correlate tool_call and tool_call_response parts using call_id when id and call_id differ."""
    import httpx
    import openai
    from agno.models.openai.responses import OpenAIResponses
    from agno.tools.function import Function

    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Sunny in {city}"

    turn = 0

    def _handler(_req: httpx.Request) -> httpx.Response:
        nonlocal turn
        turn += 1
        if turn == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_tool_turn_1",
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4o",
                    "output": [
                        {
                            "id": "fc_weather_1",
                            "call_id": "call_weather_1",
                            "type": "function_call",
                            "status": "completed",
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        }
                    ],
                    "parallel_tool_calls": True,
                    "temperature": 1.0,
                    "tool_choice": "auto",
                    "tools": [],
                    "top_p": 1.0,
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 6,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 18,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "resp_tool_turn_2",
                "object": "response",
                "created_at": 1700000001,
                "status": "completed",
                "model": "gpt-4o",
                "output": [
                    {
                        "id": "msg_2",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "It is sunny in Paris.",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "parallel_tool_calls": True,
                "temperature": 1.0,
                "tool_choice": "auto",
                "tools": [],
                "top_p": 1.0,
                "usage": {
                    "input_tokens": 20,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 8,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 28,
                },
            },
        )

    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    model = OpenAIResponses(id="gpt-4o", client=client)
    resp = model.response(
        messages=[Message(role="user", content="Weather in Paris?")],
        tools=[Function.from_callable(get_weather)],
    )
    assert resp.content == "It is sunny in Paris."

    chat_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    ]
    assert len(chat_spans) == 2
    turn1_span, turn2_span = chat_spans

    assert (
        turn1_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp_tool_turn_1"
    )
    assert (
        turn2_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp_tool_turn_2"
    )

    turn1_output: Any = turn1_span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    turn1_tool_calls = [
        part
        for msg in turn1_output
        for part in msg["parts"]
        if part.get("type") == "tool_call"
    ]
    assert len(turn1_tool_calls) == 1
    assert turn1_tool_calls[0]["id"] == "call_weather_1"
    assert turn1_tool_calls[0]["name"] == "get_weather"
    assert turn1_tool_calls[0]["arguments"] == {"city": "Paris"}

    turn2_input: Any = turn2_span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    turn2_tool_calls = [
        part
        for msg in turn2_input
        for part in msg["parts"]
        if part.get("type") == "tool_call"
    ]
    turn2_tool_responses = [
        part
        for msg in turn2_input
        for part in msg["parts"]
        if part.get("type") == "tool_call_response"
    ]
    assert len(turn2_tool_calls) == 1
    assert len(turn2_tool_responses) == 1
    assert turn2_tool_calls[0]["id"] == "call_weather_1"
    assert turn2_tool_responses[0]["id"] == "call_weather_1"
    assert turn2_tool_responses[0]["response"] == "Sunny in Paris"

    # Also verify streaming Responses tool loop correlates using call_id
    span_exporter.clear()
    stream_turn = 0

    def _stream_handler(_req: httpx.Request) -> httpx.Response:
        nonlocal stream_turn
        stream_turn += 1
        base_resp: dict[str, Any] = {
            "id": f"resp_stream_tool_{stream_turn}",
            "object": "response",
            "created_at": 1700000000 + stream_turn,
            "status": "in_progress",
            "model": "gpt-4o",
            "output": [],
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1.0,
            "usage": None,
        }
        if stream_turn == 1:
            fc_item = {
                "id": "fc_stream_weather_1",
                "call_id": "call_stream_weather_1",
                "type": "function_call",
                "status": "in_progress",
                "name": "get_weather",
                "arguments": "",
            }
            fc_done = {
                **fc_item,
                "status": "completed",
                "arguments": '{"city": "London"}',
            }
            completed = {
                **base_resp,
                "status": "completed",
                "output": [fc_done],
                "usage": {
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 5,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 15,
                },
            }
            args_delta = '{"city": "London"}'
            sse = [
                f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': base_resp})}",
                f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'sequence_number': 1, 'output_index': 0, 'item': fc_item})}",
                f"event: response.function_call_arguments.delta\ndata: {json.dumps({'type': 'response.function_call_arguments.delta', 'sequence_number': 2, 'item_id': 'fc_stream_weather_1', 'output_index': 0, 'delta': args_delta})}",
                f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'sequence_number': 3, 'output_index': 0, 'item': fc_done})}",
                f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'sequence_number': 4, 'response': completed})}",
            ]
        else:
            completed = {
                **base_resp,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_stream_2",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "It is sunny in London.",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 18,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens": 6,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 24,
                },
            }
            sse = [
                f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'sequence_number': 0, 'response': base_resp})}",
                f"event: response.output_text.delta\ndata: {json.dumps({'type': 'response.output_text.delta', 'sequence_number': 1, 'item_id': 'msg_stream_2', 'output_index': 0, 'content_index': 0, 'delta': 'It is sunny in London.', 'logprobs': []})}",
                f"event: response.completed\ndata: {json.dumps({'type': 'response.completed', 'sequence_number': 2, 'response': completed})}",
            ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=("\n\n".join(sse) + "\n\n").encode("utf-8"),
        )

    stream_client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(_stream_handler)
        ),
    )
    stream_model = OpenAIResponses(id="gpt-4o", client=stream_client)
    list(
        stream_model.response_stream(
            messages=[Message(role="user", content="Weather in London?")],
            tools=[Function.from_callable(get_weather)],
        )
    )
    stream_chat_spans = [
        s
        for s in span_exporter.get_finished_spans()
        if s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
    ]
    assert len(stream_chat_spans) == 2
    s_turn1, s_turn2 = stream_chat_spans
    assert (
        s_turn1.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp_stream_tool_1"
    )
    assert (
        s_turn2.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp_stream_tool_2"
    )
    s_turn1_calls = [
        part
        for msg in (
            s_turn1.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
            or ()
        )
        for part in msg["parts"]
        if part.get("type") == "tool_call"
    ]
    s_turn2_calls = [
        part
        for msg in (
            s_turn2.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES) or ()
        )
        for part in msg["parts"]
        if part.get("type") == "tool_call"
    ]
    s_turn2_responses = [
        part
        for msg in (
            s_turn2.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES) or ()
        )
        for part in msg["parts"]
        if part.get("type") == "tool_call_response"
    ]
    assert len(s_turn1_calls) == 1
    assert len(s_turn2_calls) == 1
    assert len(s_turn2_responses) == 1
    assert s_turn1_calls[0]["id"] == "call_stream_weather_1"
    assert s_turn2_calls[0]["id"] == "call_stream_weather_1"
    assert s_turn2_responses[0]["id"] == "call_stream_weather_1"
    assert s_turn2_responses[0]["response"] == "Sunny in London"


def test_model_stream_retry_refreshes_response_metadata(
    instrument_agno,
    span_exporter,
    metric_reader,
) -> None:
    """Partial-stream retries replace response_id and response_model with the successful retry's metadata."""
    from agno.exceptions import ModelProviderError

    class _RetryingStreamModel(MockModel):
        sync_attempts: int = 0
        async_attempts: int = 0
        id_key: str = "id"

        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            self.sync_attempts += 1
            if self.sync_attempts == 1:
                yield ModelResponse(
                    content="partial sync ",
                    provider_data={
                        self.id_key: "resp-sync-attempt-1",
                        "model": "model-sync-attempt-1",
                    },
                )
                raise ModelProviderError(
                    message="transient sync stream failure",
                    model_name=self.name,
                    model_id=self.id,
                )
            yield ModelResponse(
                provider_data={
                    self.id_key: "resp-sync-attempt-2",
                    "model": "model-sync-attempt-2",
                },
            )
            yield ModelResponse(
                content="recovered sync",
                provider_data={
                    self.id_key: "resp-sync-attempt-2",
                    "model": "model-sync-attempt-2",
                    "finish_reason": "stop",
                },
                response_usage=MessageMetrics(
                    input_tokens=11, output_tokens=7
                ),
            )

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            self.async_attempts += 1
            if self.async_attempts == 1:
                yield ModelResponse(
                    content="partial async ",
                    provider_data={
                        self.id_key: "resp-async-attempt-1",
                        "model": "model-async-attempt-1",
                    },
                )
                raise ModelProviderError(
                    message="transient async stream failure",
                    model_name=self.name,
                    model_id=self.id,
                )
            yield ModelResponse(
                provider_data={
                    self.id_key: "resp-async-attempt-2",
                    "model": "model-async-attempt-2",
                },
            )
            yield ModelResponse(
                content="recovered async",
                provider_data={
                    self.id_key: "resp-async-attempt-2",
                    "model": "model-async-attempt-2",
                    "finish_reason": "stop",
                },
                response_usage=MessageMetrics(
                    input_tokens=13, output_tokens=9
                ),
            )

    model = _RetryingStreamModel(
        id="retry-model",
        provider="OpenAI",
        retries=1,
        delay_between_retries=0,
    )

    list(model.response_stream(messages=[Message(role="user", content="go")]))

    async def _consume_async(target: _RetryingStreamModel) -> None:
        async for _ in target.aresponse_stream(
            messages=[Message(role="user", content="go async")]
        ):
            pass

    asyncio.run(_consume_async(model))

    assert model.sync_attempts == 2
    assert model.async_attempts == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    sync_span, async_span = spans

    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-sync-attempt-2"
    )
    assert (
        sync_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "model-sync-attempt-2"
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-async-attempt-2"
    )
    assert (
        async_span.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_MODEL)
        == "model-async-attempt-2"
    )

    # Also verify retry refreshes metadata when using provider_data["response_id"]
    span_exporter.clear()
    responses_retry_model = _RetryingStreamModel(
        id="retry-responses-model",
        provider="OpenAI",
        retries=1,
        delay_between_retries=0,
    )
    responses_retry_model.id_key = "response_id"
    list(
        responses_retry_model.response_stream(
            messages=[Message(role="user", content="go responses")]
        )
    )
    asyncio.run(_consume_async(responses_retry_model))
    resp_spans = span_exporter.get_finished_spans()
    assert len(resp_spans) == 2
    assert (
        resp_spans[0].attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-sync-attempt-2"
    )
    assert (
        resp_spans[1].attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
        == "resp-async-attempt-2"
    )

    metrics_data = metric_reader.get_metrics_data()
    duration_models: set[Any] = set()
    token_models: set[Any] = set()
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        duration_models.add(
                            dp.attributes.get(
                                GenAIAttributes.GEN_AI_RESPONSE_MODEL
                            )
                        )
                elif metric.name == "gen_ai.client.token.usage":
                    for dp in metric.data.data_points:
                        token_models.add(
                            dp.attributes.get(
                                GenAIAttributes.GEN_AI_RESPONSE_MODEL
                            )
                        )
    expected_models = {"model-sync-attempt-2", "model-async-attempt-2"}
    assert expected_models.issubset(duration_models)
    assert expected_models.issubset(token_models)


def test_gemini_provider_resolved_from_effective_client_settings(
    instrument_agno,
    span_exporter,
    metric_reader,
    monkeypatch,
) -> None:
    """Gemini provider resolution respects client_params and pre-built client settings without constructing a client for telemetry."""
    import httpx
    from agno.models.google import Gemini
    from google import genai
    from google.genai.types import HttpOptions
    from google.oauth2.credentials import Credentials

    for env_var in (
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_ENTERPRISE",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(env_var, raising=False)

    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Gemini!"}],
                },
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }

    vertex_urls: list[str] = []
    vertex_http = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: (
                vertex_urls.append(str(req.url)),
                httpx.Response(200, json=payload),
            )[1]
        )
    )

    real_genai_client = genai.Client
    constructed_kwargs: list[dict[str, Any]] = []

    def _spied_client(*args: Any, **kwargs: Any) -> Any:
        constructed_kwargs.append(dict(kwargs))
        return real_genai_client(*args, **kwargs)

    monkeypatch.setattr(genai, "Client", _spied_client)

    # 1. client_params={"vertexai": True, ...} overrides default vertexai=False without eagerly constructing client
    vertex_override_model = Gemini(
        id="gemini-2.5-flash",
        client_params={
            "vertexai": True,
            "project": "test-project",
            "location": "us-central1",
            "credentials": Credentials(token="test-token"),
            "http_options": {"httpx_client": vertex_http},
        },
    )
    assert resolve_model_provider(vertex_override_model) == "gcp.vertex_ai"
    assert constructed_kwargs == []
    assert vertex_override_model.client is None

    vertex_override_model.response(
        messages=[Message(role="user", content="vertex via client_params")]
    )
    assert len(constructed_kwargs) == 1
    assert constructed_kwargs[0].get("vertexai") is True
    assert any("aiplatform.googleapis.com" in u for u in vertex_urls)

    (vertex_span,) = span_exporter.get_finished_spans()
    assert (
        vertex_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "gcp.vertex_ai"
    )

    def _metric_providers() -> tuple[set[Any], set[Any]]:
        data = metric_reader.get_metrics_data()
        duration_p: set[Any] = set()
        token_p: set[Any] = set()
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == "gen_ai.client.operation.duration":
                        for dp in m.data.data_points:
                            duration_p.add(
                                dp.attributes.get(
                                    GenAIAttributes.GEN_AI_PROVIDER_NAME
                                )
                            )
                    elif m.name == "gen_ai.client.token.usage":
                        for dp in m.data.data_points:
                            token_p.add(
                                dp.attributes.get(
                                    GenAIAttributes.GEN_AI_PROVIDER_NAME
                                )
                            )
        return duration_p, token_p

    duration_providers, token_providers = _metric_providers()
    assert duration_providers == {"gcp.vertex_ai"}
    assert token_providers == {"gcp.vertex_ai"}

    # 2. client_params={"vertexai": False, ...} overrides vertexai=True
    span_exporter.clear()
    gemini_urls: list[str] = []
    gemini_http = httpx.Client(
        transport=httpx.MockTransport(
            lambda req: (
                gemini_urls.append(str(req.url)),
                httpx.Response(200, json=payload),
            )[1]
        )
    )
    gemini_override_model = Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        client_params={
            "vertexai": False,
            "api_key": "test-key",
            "project": None,
            "location": None,
            "http_options": {"httpx_client": gemini_http},
        },
    )
    assert resolve_model_provider(gemini_override_model) == "gcp.gemini"
    assert len(constructed_kwargs) == 1
    gemini_override_model.response(
        messages=[Message(role="user", content="gemini via client_params")]
    )
    assert len(constructed_kwargs) == 2
    assert any("generativelanguage.googleapis.com" in u for u in gemini_urls)
    (gemini_span,) = span_exporter.get_finished_spans()
    assert (
        gemini_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "gcp.gemini"
    )

    # 3. Pre-built genai.Client(vertexai=True, ...) passed directly overrides default vertexai=False
    span_exporter.clear()
    prebuilt_vertex_client = real_genai_client(
        vertexai=True,
        project="test-project",
        location="us-central1",
        credentials=Credentials(token="test-token"),
        http_options=HttpOptions(httpx_client=vertex_http),
    )
    prebuilt_model = Gemini(
        id="gemini-2.5-flash",
        client=prebuilt_vertex_client,
    )
    prebuilt_model.response(
        messages=[Message(role="user", content="vertex via prebuilt client")]
    )
    (prebuilt_span,) = span_exporter.get_finished_spans()
    assert (
        prebuilt_span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == "gcp.vertex_ai"
    )

    # 4. client_params={"vertexai": None} defers to env var instead of adapter flag
    none_override_model = Gemini(
        id="gemini-2.5-flash",
        vertexai=True,
        client_params={"vertexai": None},
    )
    assert resolve_model_provider(none_override_model) == "gcp.gemini"
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    assert resolve_model_provider(none_override_model) == "gcp.vertex_ai"


def test_model_endpoint_omits_default_scheme_ports(
    instrument_agno,
    span_exporter,
    metric_reader,
) -> None:
    """Explicit default ports (https:443, http:80) are omitted from server.port while non-default ports are preserved."""
    import httpx
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-default-port",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }

    def _make_http_client() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        )

    https_default_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://api.example.com:443/v1",
        http_client=_make_http_client(),
    )
    https_default_model.response(messages=[Message(role="user", content="hi")])

    http_default_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="http://plain.example.com:80/v1",
        http_client=_make_http_client(),
    )
    http_default_model.response(messages=[Message(role="user", content="hi")])

    custom_port_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://custom.example.com:8443/v1",
        http_client=_make_http_client(),
    )
    custom_port_model.response(messages=[Message(role="user", content="hi")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    https_span, http_span, custom_span = spans

    assert https_span.attributes.get("server.address") == "api.example.com"
    assert "server.port" not in https_span.attributes

    assert http_span.attributes.get("server.address") == "plain.example.com"
    assert "server.port" not in http_span.attributes

    assert custom_span.attributes.get("server.address") == "custom.example.com"
    assert custom_span.attributes.get("server.port") == 8443

    metrics_data = metric_reader.get_metrics_data()
    observed_ports_by_addr: dict[str, set[Any]] = {}
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        addr = str(dp.attributes.get("server.address"))
                        observed_ports_by_addr.setdefault(addr, set()).add(
                            dp.attributes.get("server.port")
                        )
    assert observed_ports_by_addr["api.example.com"] == {None}
    assert observed_ports_by_addr["plain.example.com"] == {None}
    assert observed_ports_by_addr["custom.example.com"] == {8443}


def test_model_stream_abandoned_after_caller_error_in_bare_iteration(
    instrument_agno,
    span_exporter,
    metric_reader,
    vcr,
) -> None:
    """Abandoning a partially read public response_stream/aresponse_stream after a bare-iteration caller error ends the span with ERROR and error.type."""
    import gc

    import openai
    from agno.models.openai import OpenAIChat

    from opentelemetry.semconv.attributes.error_attributes import (
        ErrorTypeValues,
    )

    sync_yielded = 0
    async_yielded = 0

    class _MultiChunkModel(MockModel):
        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            nonlocal sync_yielded
            sync_yielded += 1
            yield ModelResponse(content="chunk 1")
            sync_yielded += 1
            yield ModelResponse(content="chunk 2")

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            nonlocal async_yielded
            async_yielded += 1
            yield ModelResponse(content="async chunk 1")
            async_yielded += 1
            yield ModelResponse(content="async chunk 2")

    model = _MultiChunkModel(id="abandon-stream-model", provider="OpenAI")

    def _run_sync_bare_error() -> None:
        stream = model.response_stream(
            messages=[Message(role="user", content="hi")]
        )
        with pytest.raises(RuntimeError, match="caller sync failure"):
            for chunk in stream:
                if getattr(chunk, "content", None):
                    raise RuntimeError("caller sync failure")
        # Bare iteration does not run context-manager exit; span finalizes only on abandonment.
        assert sync_yielded == 1
        assert not span_exporter.get_finished_spans()
        del stream
        gc.collect()
        gc.collect()

    _run_sync_bare_error()
    gc.collect()
    gc.collect()

    sync_spans = span_exporter.get_finished_spans()
    assert len(sync_spans) == 1
    assert sync_spans[0].status.status_code == StatusCode.ERROR
    assert sync_spans[0].status.description == "abandoned stream"
    assert sync_spans[0].attributes.get(ERROR_TYPE) == (
        ErrorTypeValues.OTHER.value
    )
    assert (
        sync_spans[0].attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM)
        is True
    )

    span_exporter.clear()

    async def _run_async_bare_error() -> None:
        stream = model.aresponse_stream(
            messages=[Message(role="user", content="hi async")]
        )
        with pytest.raises(RuntimeError, match="caller async failure"):
            async for chunk in stream:
                if getattr(chunk, "content", None):
                    raise RuntimeError("caller async failure")
        # Bare async iteration does not run __aexit__; span finalizes only on abandonment.
        assert async_yielded == 1
        assert not span_exporter.get_finished_spans()
        del stream
        gc.collect()
        gc.collect()

    asyncio.run(_run_async_bare_error())
    gc.collect()
    gc.collect()

    async_spans = span_exporter.get_finished_spans()
    assert len(async_spans) == 1
    assert async_spans[0].status.status_code == StatusCode.ERROR
    assert async_spans[0].status.description == "abandoned stream"
    assert async_spans[0].attributes.get(ERROR_TYPE) == (
        ErrorTypeValues.OTHER.value
    )
    assert (
        async_spans[0].attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM)
        is True
    )

    # Also verify abandoning partially read recorded public OpenAIChat sync/async streams
    prompt = [
        Message(
            role="user",
            content="A farmer has 17 sheep and all but 9 run away. How many sheep remain? Explain your reasoning.",
        )
    ]

    span_exporter.clear()

    def _run_recorded_sync_abandon() -> None:
        chat_model = OpenAIChat(
            id="gpt-5.1",
            client=openai.OpenAI(api_key="test_openai_api_key"),
        )
        with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
            stream = chat_model.response_stream(messages=prompt)
            with pytest.raises(
                RuntimeError, match="recorded sync caller error"
            ):
                for chunk in stream:
                    if getattr(chunk, "content", None):
                        raise RuntimeError("recorded sync caller error")
            assert not span_exporter.get_finished_spans()
            del stream
            gc.collect()
            gc.collect()

    _run_recorded_sync_abandon()
    gc.collect()
    gc.collect()

    (rec_sync_span,) = span_exporter.get_finished_spans()
    assert rec_sync_span.status.status_code == StatusCode.ERROR
    assert rec_sync_span.status.description == "abandoned stream"
    assert rec_sync_span.attributes.get(ERROR_TYPE) == (
        ErrorTypeValues.OTHER.value
    )
    assert (
        rec_sync_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM)
        is True
    )

    span_exporter.clear()

    async def _run_recorded_async_abandon() -> None:
        async_chat_model = OpenAIChat(
            id="gpt-5.1",
            async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
        )
        stream = async_chat_model.aresponse_stream(messages=prompt)
        with pytest.raises(RuntimeError, match="recorded async caller error"):
            async for chunk in stream:
                if getattr(chunk, "content", None):
                    raise RuntimeError("recorded async caller error")
        assert not span_exporter.get_finished_spans()
        del stream
        gc.collect()
        gc.collect()

    with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
        asyncio.run(_run_recorded_async_abandon())
    gc.collect()
    gc.collect()

    (rec_async_span,) = span_exporter.get_finished_spans()
    assert rec_async_span.status.status_code == StatusCode.ERROR
    assert rec_async_span.status.description == "abandoned stream"
    assert rec_async_span.attributes.get(ERROR_TYPE) == (
        ErrorTypeValues.OTHER.value
    )
    assert (
        rec_async_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STREAM)
        is True
    )

    metrics_data = metric_reader.get_metrics_data()
    duration_counts_by_error_type: dict[Any, int] = {}
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        err = dp.attributes.get(ERROR_TYPE)
                        duration_counts_by_error_type[err] = (
                            duration_counts_by_error_type.get(err, 0)
                            + dp.count
                        )
    assert duration_counts_by_error_type == {ErrorTypeValues.OTHER.value: 4}


def test_model_async_stream_context_manager_preserves_body_error_when_delegate_cleanup_cancelled(
    instrument_agno,
    span_exporter,
) -> None:
    """When async with model.aresponse_stream(...) body raises ValueError and delegate cleanup raises CancelledError, ValueError is preserved and all delegates close."""
    from agno.models.base import Model

    events: list[str] = []

    class _BaseCancellingDelegateModel(Model):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="unused")

        def invoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> Iterator[ModelResponse]:
            yield ModelResponse(content="unused")

        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                yield ModelResponse(content="chunk1")
                yield ModelResponse(content="chunk2")
            finally:
                await asyncio.sleep(0)
                events.append("base_cancelled")
                raise asyncio.CancelledError("delegate close cancelled")

        def _parse_provider_response(
            self, response: Any, **kwargs: Any
        ) -> ModelResponse:
            return response

        def _parse_provider_response_delta(
            self, response: Any
        ) -> ModelResponse:
            return response

    class _OuterDelegateModel(_BaseCancellingDelegateModel):
        async def ainvoke_stream(
            self, *args: Any, **kwargs: Any
        ) -> AsyncIterator[ModelResponse]:
            try:
                async for delta in super().ainvoke_stream(*args, **kwargs):
                    yield delta
            finally:
                await asyncio.sleep(0)
                events.append("outer_closed")

    async def _run() -> None:
        # 1. Body raises ValueError while inner delegate cleanup raises CancelledError:
        #    ValueError must survive and every delegate must still close.
        model = _OuterDelegateModel(id="ctx-cancel-body", provider="Custom")
        with pytest.raises(ValueError, match="caller body failure"):
            async with model.aresponse_stream(
                messages=[Message(role="user", content="hi")]
            ) as stream:
                async for chunk in stream:
                    if getattr(chunk, "content", None):
                        raise ValueError("caller body failure")
        assert events == ["base_cancelled", "outer_closed"]

        # 2. Normal exit (no body exception) still propagates CancelledError from cleanup
        #    and closes all delegates.
        events.clear()
        model2 = _OuterDelegateModel(id="ctx-cancel-normal", provider="Custom")
        with pytest.raises(
            asyncio.CancelledError, match="delegate close cancelled"
        ):
            async with model2.aresponse_stream(
                messages=[Message(role="user", content="hi")]
            ) as stream2:
                async for chunk in stream2:
                    if getattr(chunk, "content", None):
                        break
        assert events == ["base_cancelled", "outer_closed"]

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes.get(ERROR_TYPE) == "ValueError"
    assert spans[1].status.status_code == StatusCode.ERROR
    assert (
        spans[1].attributes.get(ERROR_TYPE)
        == "asyncio.exceptions.CancelledError"
    )


def test_recorded_openai_chat_streaming_and_cached_tokens(
    instrument_agno_content_capture,
    span_exporter,
    vcr,
) -> None:
    """Replay recorded OpenAI Chat Completions responses through real OpenAIChat sync/async streaming and cached-token paths."""
    import openai
    from agno.models.openai import OpenAIChat

    # 1. Sync and async streaming replay with real reasoning tokens
    with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
        sync_model = OpenAIChat(
            id="gpt-5.1",
            client=openai.OpenAI(api_key="test_openai_api_key"),
        )
        sync_chunks = list(
            sync_model.response_stream(
                messages=[
                    Message(
                        role="user",
                        content="A farmer has 17 sheep and all but 9 run away. How many sheep remain? Explain your reasoning.",
                    )
                ]
            )
        )
    assert any(getattr(c, "content", None) for c in sync_chunks)

    async def _run_async_chat_stream() -> list[Any]:
        async_model = OpenAIChat(
            id="gpt-5.1",
            async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
        )
        collected: list[Any] = []
        async for chunk in async_model.aresponse_stream(
            messages=[
                Message(
                    role="user",
                    content="A farmer has 17 sheep and all but 9 run away. How many sheep remain? Explain your reasoning.",
                )
            ]
        ):
            collected.append(chunk)
        return collected

    with vcr.use_cassette("openai_chat_streaming_reasoning.yaml"):
        async_chunks = asyncio.run(_run_async_chat_stream())
    assert any(getattr(c, "content", None) for c in async_chunks)

    emits_response_id = any(
        isinstance(getattr(c, "provider_data", None), dict)
        and c.provider_data.get("id")
        for c in sync_chunks
    )
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        attrs = span.attributes
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL) == "gpt-5.1"
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 29
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 92
        assert (
            attrs.get(GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)
            == 10
        )
        assert attrs.get("server.address") == "api.openai.com"
        assert "server.port" not in attrs
        if emits_response_id:
            assert (
                attrs.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
                == "chatcmpl-EMxk4dQaiUamioadk6bK1Y462yRui"
            )
        output_msgs: Any = attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert (
            output_msgs
            and "9 sheep remain" in output_msgs[0]["parts"][0]["content"]
        )

    # 2. Recorded sync + async non-streaming and streaming cached prompt tokens
    span_exporter.clear()
    with vcr.use_cassette("openai_chat_cached_tokens.yaml"):
        cached_model = OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=8,
            client=openai.OpenAI(api_key="test_openai_api_key"),
        )
        non_stream_resp = cached_model.response(
            messages=[Message(role="user", content="Reply with OK only.")]
        )
        stream_chunks = list(
            cached_model.response_stream(
                messages=[Message(role="user", content="Reply with OK only.")]
            )
        )
    assert non_stream_resp.content == "OK"
    assert any(getattr(c, "content", None) for c in stream_chunks)

    async def _run_async_cached() -> tuple[Any, list[Any]]:
        async_cached_model = OpenAIChat(
            id="gpt-4o-mini",
            max_tokens=8,
            async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
        )
        async_non_stream = await async_cached_model.aresponse(
            messages=[Message(role="user", content="Reply with OK only.")]
        )
        async_stream: list[Any] = []
        async for chunk in async_cached_model.aresponse_stream(
            messages=[Message(role="user", content="Reply with OK only.")]
        ):
            async_stream.append(chunk)
        return async_non_stream, async_stream

    with vcr.use_cassette("openai_chat_cached_tokens.yaml"):
        async_non_stream_resp, async_stream_chunks = asyncio.run(
            _run_async_cached()
        )
    assert async_non_stream_resp.content == "OK"
    assert any(getattr(c, "content", None) for c in async_stream_chunks)

    cached_spans = span_exporter.get_finished_spans()
    assert len(cached_spans) == 4
    for span, expected_stream in zip(
        cached_spans, (None, True, None, True), strict=True
    ):
        attrs = span.attributes
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 1457
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 2
        assert (
            attrs.get(GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS)
            == 1408
        )
        assert (
            attrs.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) == expected_stream
        )
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 8
        assert attrs.get("server.address") == "api.openai.com"
        assert "server.port" not in attrs


def test_recorded_openai_responses_streaming_and_tool_calling(
    instrument_agno_content_capture,
    span_exporter,
    vcr,
) -> None:
    """Replay recorded OpenAI Responses sync/async streams and two-turn tool calls through real OpenAIResponses."""
    import openai
    from agno.models.openai.responses import OpenAIResponses
    from agno.tools.function import Function

    # 1. Sync and async streaming replay
    with vcr.use_cassette("openai_responses_streaming.yaml"):
        sync_model = OpenAIResponses(
            id="gpt-4o-mini",
            client=openai.OpenAI(api_key="test_openai_api_key"),
        )
        list(
            sync_model.response_stream(
                messages=[Message(role="user", content="Say this is a test")]
            )
        )

    async def _run_async_responses_stream() -> None:
        async_model = OpenAIResponses(
            id="gpt-4o-mini",
            async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
        )
        async for _ in async_model.aresponse_stream(
            messages=[Message(role="user", content="Say this is a test")]
        ):
            pass

    with vcr.use_cassette("openai_responses_streaming.yaml"):
        asyncio.run(_run_async_responses_stream())

    stream_spans = span_exporter.get_finished_spans()
    assert len(stream_spans) == 2
    for span in stream_spans:
        attrs = span.attributes
        assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "openai"
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        assert (
            attrs.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "resp_048385de343c399f006a2e1ac3c6e881a0b3e7d86be0a3aa51"
        )
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 22
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 6
        assert attrs.get("server.address") == "api.openai.com"
        assert "server.port" not in attrs
        output_msgs: Any = attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert output_msgs[0]["parts"][0]["content"] == "This is a test."

    # 2. Recorded sync + async two-turn Responses tool loop with distinct fc_* and call_* IDs
    def get_current_weather(location: str) -> str:
        """Get the current weather in a given location."""
        if "Seattle" in location:
            return '{"location":"Seattle, WA","temperature":58,"unit":"F","conditions":"rain"}'
        return '{"location":"San Francisco, CA","temperature":65,"unit":"F","conditions":"fog"}'

    expected_call_ids = [
        "call_krAHY11uDHrSPJIjNZLF7RHU",
        "call_fr3gmaw1ffbKMC3N64V6Iux4",
    ]

    def _assert_two_turn_tool_spans(finished_spans: Any) -> None:
        chat_spans = [
            s
            for s in finished_spans
            if s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "chat"
        ]
        tool_spans = [
            s
            for s in finished_spans
            if s.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
            == "execute_tool"
        ]
        assert len(chat_spans) == 2
        assert len(tool_spans) == 2
        turn1, turn2 = chat_spans
        assert (
            turn1.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "resp_06846aae2b482d8e006a9b1773c4ec87d08182937f685f261f"
        )
        assert (
            turn2.attributes.get(GenAIAttributes.GEN_AI_RESPONSE_ID)
            == "resp_06846aae2b482d8e006a9b1775645c87d08005ec0fb2110764"
        )
        for span in (turn1, turn2):
            assert span.attributes.get("server.address") == "api.openai.com"
            assert "server.port" not in span.attributes
        turn1_call_ids = [
            part["id"]
            for msg in (
                turn1.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
                or ()
            )
            for part in msg["parts"]
            if part.get("type") == "tool_call"
        ]
        turn2_call_ids = [
            part["id"]
            for msg in (
                turn2.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
                or ()
            )
            for part in msg["parts"]
            if part.get("type") == "tool_call"
        ]
        turn2_result_ids = [
            part["id"]
            for msg in (
                turn2.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
                or ()
            )
            for part in msg["parts"]
            if part.get("type") == "tool_call_response"
        ]
        assert turn1_call_ids == expected_call_ids
        assert turn2_call_ids == expected_call_ids
        assert turn2_result_ids == expected_call_ids
        assert all(
            isinstance(
                s.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID), str
            )
            and s.attributes.get(GenAIAttributes.GEN_AI_TOOL_CALL_ID)
            for s in tool_spans
        )
        turn2_outputs: Any = turn2.attributes.get(
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        )
        assert turn2_outputs and "Seattle, WA" in str(
            turn2_outputs[0]["parts"][0]["content"]
        )

    span_exporter.clear()
    with vcr.use_cassette("openai_responses_tool_calling.yaml"):
        tool_model = OpenAIResponses(
            id="gpt-4o-mini",
            client=openai.OpenAI(api_key="test_openai_api_key"),
        )
        resp = tool_model.response(
            messages=[
                Message(
                    role="user",
                    content="What's the weather in Seattle and San Francisco today?",
                )
            ],
            tools=[Function.from_callable(get_current_weather)],
        )
    assert "Seattle, WA" in str(resp.content)
    _assert_two_turn_tool_spans(span_exporter.get_finished_spans())

    span_exporter.clear()

    async def _run_async_responses_tools() -> Any:
        async_tool_model = OpenAIResponses(
            id="gpt-4o-mini",
            async_client=openai.AsyncOpenAI(api_key="test_openai_api_key"),
        )
        return await async_tool_model.aresponse(
            messages=[
                Message(
                    role="user",
                    content="What's the weather in Seattle and San Francisco today?",
                )
            ],
            tools=[Function.from_callable(get_current_weather)],
        )

    with vcr.use_cassette("openai_responses_tool_calling.yaml"):
        async_resp = asyncio.run(_run_async_responses_tools())
    assert "Seattle, WA" in str(async_resp.content)
    _assert_two_turn_tool_spans(span_exporter.get_finished_spans())


def test_recorded_gemini_generate_content(
    instrument_agno_content_capture,
    span_exporter,
    vcr,
    monkeypatch,
) -> None:
    """Replay a recorded Google Gemini generateContent response through real sync and async Gemini paths."""
    from agno.models.google import Gemini
    from google import genai
    from google.genai.types import HttpOptions

    for env_var in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(env_var, raising=False)

    with vcr.use_cassette("gemini_generate_content.yaml"):
        model = Gemini(
            id="gemini-2.5-flash",
            client=genai.Client(
                api_key="test_google_genai_api_key",
                http_options=HttpOptions(api_version="v1beta"),
            ),
        )
        resp = model.response(
            messages=[Message(role="user", content="Say this is a test")]
        )
    assert resp.content == "This is a test."

    async def _run_async_gemini() -> Any:
        async_model = Gemini(
            id="gemini-2.5-flash",
            client=genai.Client(
                api_key="test_google_genai_api_key",
                http_options=HttpOptions(api_version="v1beta"),
            ),
        )
        return await async_model.aresponse(
            messages=[Message(role="user", content="Say this is a test")]
        )

    with vcr.use_cassette("gemini_generate_content.yaml"):
        async_resp = asyncio.run(_run_async_gemini())
    assert async_resp.content == "This is a test."

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span, call_resp in zip(spans, (resp, async_resp), strict=True):
        attrs = span.attributes
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "gcp.gemini"
        assert (
            attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
            == "gemini-2.5-flash"
        )
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 6
        assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 25
        if getattr(call_resp.response_usage, "reasoning_tokens", 0):
            assert (
                attrs.get(GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)
                == 20
            )
        assert (
            attrs.get("server.address") == "generativelanguage.googleapis.com"
        )
        assert "server.port" not in attrs
        output_msgs: Any = attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert output_msgs[0]["parts"][0]["content"] == "This is a test."


def test_model_endpoint_honors_client_params_base_url_during_lazy_init(
    instrument_agno,
    span_exporter,
    metric_reader,
) -> None:
    """client_params base_url overrides adapter base_url before lazy sync/async client init without eager construction."""
    import httpx
    from agno.models.google import Gemini
    from agno.models.openai import OpenAIChat

    payload = {
        "id": "chatcmpl-client-params",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
        },
    }

    requested_hosts: list[tuple[str, int | None]] = []

    def _sync_handler(req: httpx.Request) -> httpx.Response:
        requested_hosts.append((req.url.host, req.url.port))
        return httpx.Response(200, json=payload)

    async def _async_handler(req: httpx.Request) -> httpx.Response:
        requested_hosts.append((req.url.host, req.url.port))
        return httpx.Response(200, json=payload)

    # 1. Lazy sync initialization: client is None before call 1, cached on call 2
    sync_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://adapter.example.com/v1",
        client_params={"base_url": "https://override.example.com:8443/v1"},
        http_client=httpx.Client(transport=httpx.MockTransport(_sync_handler)),
    )
    assert sync_model.client is None
    sync_model.response(messages=[Message(role="user", content="first sync")])
    assert sync_model.client is not None
    sync_model.response(messages=[Message(role="user", content="second sync")])

    # 2. Lazy async initialization: async_client is None before call 1, cached on call 2
    async_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://adapter.example.com/v1",
        client_params={
            "base_url": "https://async-override.example.com:9443/v1"
        },
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(_async_handler)
        ),
    )

    async def _run_lazy_async() -> None:
        assert async_model.async_client is None
        await async_model.aresponse(
            messages=[Message(role="user", content="first async")]
        )
        assert async_model.async_client is not None
        await async_model.aresponse(
            messages=[Message(role="user", content="second async")]
        )

    asyncio.run(_run_lazy_async())

    # 3. Default port in client_params base_url is normalized out; explicit None suppresses adapter base_url
    default_port_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://adapter.example.com:8080/v1",
        client_params={"base_url": "https://default-port.example.com:443/v1"},
        http_client=httpx.Client(transport=httpx.MockTransport(_sync_handler)),
    )
    default_port_model.response(
        messages=[Message(role="user", content="default port")]
    )

    cleared_base_url_model = OpenAIChat(
        id="gpt-4o",
        api_key="test-key",
        base_url="https://adapter.example.com/v1",
        client_params={"base_url": None},
        http_client=httpx.Client(transport=httpx.MockTransport(_sync_handler)),
    )
    cleared_base_url_model.response(
        messages=[Message(role="user", content="cleared base url")]
    )

    # 4. Gemini client_params.http_options.base_url before lazy client construction
    gemini_payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 3,
            "candidatesTokenCount": 2,
            "totalTokenCount": 5,
        },
    }
    gemini_model = Gemini(
        id="gemini-2.5-flash",
        api_key="test-key",
        client_params={
            "http_options": {
                "base_url": "https://gemini-proxy.example.com:8443",
                "httpx_client": httpx.Client(
                    transport=httpx.MockTransport(
                        lambda _req: httpx.Response(200, json=gemini_payload)
                    )
                ),
            }
        },
    )
    assert gemini_model.client is None
    gemini_model.response(messages=[Message(role="user", content="hi")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 7
    sync1, sync2, async1, async2, default_port_span, cleared_span, gem_span = (
        spans
    )

    for span in (sync1, sync2):
        assert span.attributes.get("server.address") == "override.example.com"
        assert span.attributes.get("server.port") == 8443
    for span in (async1, async2):
        assert (
            span.attributes.get("server.address")
            == "async-override.example.com"
        )
        assert span.attributes.get("server.port") == 9443
    assert (
        default_port_span.attributes.get("server.address")
        == "default-port.example.com"
    )
    assert "server.port" not in default_port_span.attributes
    assert "server.address" not in cleared_span.attributes
    assert "server.port" not in cleared_span.attributes
    assert (
        gem_span.attributes.get("server.address") == "gemini-proxy.example.com"
    )
    assert gem_span.attributes.get("server.port") == 8443

    assert requested_hosts[:4] == [
        ("override.example.com", 8443),
        ("override.example.com", 8443),
        ("async-override.example.com", 9443),
        ("async-override.example.com", 9443),
    ]

    metrics_data = metric_reader.get_metrics_data()
    counts_by_endpoint: dict[tuple[Any, Any], int] = {}
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        key = (
                            dp.attributes.get("server.address"),
                            dp.attributes.get("server.port"),
                        )
                        counts_by_endpoint[key] = (
                            counts_by_endpoint.get(key, 0) + dp.count
                        )
    assert counts_by_endpoint[("override.example.com", 8443)] == 2
    assert counts_by_endpoint[("async-override.example.com", 9443)] == 2
    assert counts_by_endpoint[("default-port.example.com", None)] == 1
    assert counts_by_endpoint[("gemini-proxy.example.com", 8443)] == 1
    assert all(addr != "adapter.example.com" for addr, _ in counts_by_endpoint)


def test_model_bedrock_tagged_content_blocks(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Bedrock Converse tagged content blocks (image, video, document, reasoningContent, toolUse, toolResult) are captured via public Model API."""
    model = _SyncModel(
        id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        provider="AwsBedrock",
    )
    png_bytes = b"\x89PNG\r\n\x1a\nfake-png"
    pdf_bytes = b"%PDF-1.4\nfake-pdf"
    mp4_bytes = b"\x00\x00\x00\x18ftypmp42"

    # 1. Image-only Bedrock input must produce a BlobPart, not an empty TextPart
    model.response(
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": png_bytes},
                        }
                    }
                ],
            )
        ]
    )
    (img_only_span,) = span_exporter.get_finished_spans()
    img_only_msgs: Any = img_only_span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert len(img_only_msgs) == 1
    assert list(img_only_msgs[0]["parts"]) == [
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": png_bytes,
        }
    ]

    # 2. Mixed Bedrock blocks across multi-turn conversation: text, image, video,
    #    document (bytes, text, s3Location), reasoningContent, toolUse, toolResult
    span_exporter.clear()
    model.response(
        messages=[
            Message(
                role="user",
                content=[
                    {"text": "Review these attachments"},
                    {
                        "image": {
                            "format": "png",
                            "source": {"bytes": png_bytes},
                        }
                    },
                    {
                        "video": {
                            "format": "mp4",
                            "source": {"bytes": mp4_bytes},
                        }
                    },
                    {
                        "document": {
                            "format": "pdf",
                            "name": "spec",
                            "source": {"bytes": pdf_bytes},
                        }
                    },
                    {
                        "document": {
                            "format": "txt",
                            "name": "inline",
                            "source": {"text": "inline doc text"},
                        }
                    },
                    {
                        "document": {
                            "format": "csv",
                            "name": "metrics",
                            "source": {
                                "s3Location": {
                                    "uri": "s3://bucket/metrics.csv"
                                }
                            },
                        }
                    },
                ],
            ),
            Message(
                role="assistant",
                content=[
                    {
                        "reasoningContent": {
                            "reasoningText": {"text": "Checking weather first"}
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_abc",
                            "name": "get_weather",
                            "input": {"city": "Seattle"},
                        }
                    },
                ],
            ),
            Message(
                role="user",
                content=[
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_abc",
                            "content": [{"json": {"temp_f": 58}}],
                        }
                    }
                ],
            ),
        ]
    )

    (mixed_span,) = span_exporter.get_finished_spans()
    input_msgs: Any = mixed_span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert len(input_msgs) == 3

    assert list(input_msgs[0]["parts"]) == [
        {"type": "text", "content": "Review these attachments"},
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": png_bytes,
        },
        {
            "type": "blob",
            "modality": "video",
            "mime_type": "video/mp4",
            "content": mp4_bytes,
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": pdf_bytes,
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "text/plain",
            "content": b"inline doc text",
        },
        {
            "type": "uri",
            "modality": "document",
            "mime_type": "text/csv",
            "uri": "s3://bucket/metrics.csv",
        },
    ]
    assert list(input_msgs[1]["parts"]) == [
        {"type": "reasoning", "content": "Checking weather first"},
        {
            "type": "tool_call",
            "id": "tooluse_abc",
            "name": "get_weather",
            "arguments": {"city": "Seattle"},
        },
    ]
    assert list(input_msgs[2]["parts"]) == [
        {
            "type": "tool_call_response",
            "id": "tooluse_abc",
            "response": ({"json": {"temp_f": 58}},),
        }
    ]


def test_recorded_gemini_stream_generate_content(
    instrument_agno_content_capture,
    span_exporter,
    metric_reader,
    vcr,
    monkeypatch,
) -> None:
    """Replay a recorded multi-chunk Google Gemini streamGenerateContent response through sync and async Gemini streams."""
    from agno.models.google import Gemini
    from google import genai
    from google.genai.types import HttpOptions

    for env_var in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
    ):
        monkeypatch.delenv(env_var, raising=False)

    prompt = [
        Message(
            role="user",
            content="Write a short 3-sentence summary of why OpenTelemetry is useful.",
        )
    ]

    with vcr.use_cassette("gemini_stream_generate_content.yaml"):
        sync_model = Gemini(
            id="gemini-2.5-flash",
            client=genai.Client(
                api_key="test_google_genai_api_key",
                http_options=HttpOptions(api_version="v1beta"),
            ),
            temperature=0.2,
            max_output_tokens=1024,
        )
        sync_chunks = list(sync_model.response_stream(messages=prompt))
    assert any(getattr(c, "content", None) for c in sync_chunks)

    async def _run_async_gemini_stream() -> list[Any]:
        async_model = Gemini(
            id="gemini-2.5-flash",
            client=genai.Client(
                api_key="test_google_genai_api_key",
                http_options=HttpOptions(api_version="v1beta"),
            ),
            temperature=0.2,
            max_output_tokens=1024,
        )
        collected: list[Any] = []
        async for chunk in async_model.aresponse_stream(messages=prompt):
            collected.append(chunk)
        return collected

    with vcr.use_cassette("gemini_stream_generate_content.yaml"):
        async_chunks = asyncio.run(_run_async_gemini_stream())
    assert any(getattr(c, "content", None) for c in async_chunks)

    import importlib.metadata

    is_agno_v2 = importlib.metadata.version("agno").startswith("2.")
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        attrs = span.attributes
        assert attrs.get(GenAIAttributes.GEN_AI_OPERATION_NAME) == "chat"
        assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "gcp.gemini"
        assert (
            attrs.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)
            == "gemini-2.5-flash"
        )
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_STREAM) is True
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.2
        assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 1024
        if is_agno_v2:
            # Agno 2.3.x sums per-chunk cumulative Gemini usageMetadata across
            # the 3 streamed SSE chunks rather than taking the final cumulative value.
            assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 48
            assert (
                attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 2538
            )
        else:
            assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 16
            assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 870
            assert (
                attrs.get(GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS)
                == 800
            )
        assert (
            attrs.get("server.address") == "generativelanguage.googleapis.com"
        )
        assert "server.port" not in attrs
        output_msgs: Any = attrs.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
        assert (
            output_msgs
            and "OpenTelemetry provides"
            in output_msgs[0]["parts"][0]["content"]
        )

    metric_names = {
        metric.name
        for rm in metric_reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }
    assert "gen_ai.client.operation.time_to_first_chunk" in metric_names
    assert "gen_ai.client.operation.time_per_output_chunk" in metric_names


def test_gemini_generation_config_and_generative_model_kwargs_precedence(
    instrument_agno,
    span_exporter,
) -> None:
    """Gemini generation_config and generative_model_kwargs are honored with Agno's override precedence."""
    import json

    import httpx
    from agno.models.google import Gemini
    from google import genai
    from google.genai.types import GenerateContentConfig, HttpOptions

    captured_gen_configs: list[dict[str, Any]] = []
    gemini_payload = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 4,
            "totalTokenCount": 14,
        },
    }

    def _sync_handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        captured_gen_configs.append(body.get("generationConfig") or {})
        return httpx.Response(200, json=gemini_payload)

    async def _async_handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content.decode("utf-8"))
        captured_gen_configs.append(body.get("generationConfig") or {})
        return httpx.Response(200, json=gemini_payload)

    def _make_client() -> genai.Client:
        return genai.Client(
            api_key="test-key",
            http_options=HttpOptions(
                httpx_client=httpx.Client(
                    transport=httpx.MockTransport(_sync_handler)
                ),
                httpx_async_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(_async_handler)
                ),
            ),
        )

    # 1. Dict generation_config
    dict_cfg_model = Gemini(
        id="gemini-2.5-flash",
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 128,
            "top_p": 0.8,
            "top_k": 32,
            "stop_sequences": ["END"],
            "seed": 9,
        },
        client=_make_client(),
    )
    dict_cfg_model.response(
        messages=[Message(role="user", content="dict cfg")]
    )
    (dict_span,) = span_exporter.get_finished_spans()
    assert (
        dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
        == 0.3
    )
    assert (
        dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS)
        == 128
    )
    assert (
        dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.8
    )
    assert dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_K) == 32
    assert list(
        dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES)
        or ()
    ) == ["END"]
    assert dict_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_SEED) == 9
    # On Agno >= 3.2, verify outgoing wire generationConfig matches telemetry
    last_cfg = captured_gen_configs[-1]
    if "temperature" in last_cfg:
        assert last_cfg["temperature"] == 0.3
        assert last_cfg["maxOutputTokens"] == 128

    span_exporter.clear()

    # 2. GenerateContentConfig object in generation_config (async path)
    typed_cfg_model = Gemini(
        id="gemini-2.5-flash",
        generation_config=GenerateContentConfig(
            temperature=0.35,
            max_output_tokens=200,
            top_k=24,
        ),
        client=_make_client(),
    )
    asyncio.run(
        typed_cfg_model.aresponse(
            messages=[Message(role="user", content="typed cfg")]
        )
    )
    (typed_span,) = span_exporter.get_finished_spans()
    assert (
        typed_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
        == 0.35
    )
    assert (
        typed_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS)
        == 200
    )
    assert (
        typed_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_K) == 24
    )

    span_exporter.clear()

    # 3. Override precedence: direct attrs > generative_model_kwargs > generation_config
    #    (and explicit None in generative_model_kwargs clears generation_config).
    precedence_model = Gemini(
        id="gemini-2.5-flash",
        generation_config={
            "temperature": 0.3,
            "top_p": 0.6,
            "max_output_tokens": 128,
            "top_k": 20,
        },
        generative_model_kwargs={
            "temperature": 0.5,
            "top_p": 0.75,
            "top_k": None,
        },
        temperature=0.8,
        client=_make_client(),
    )
    precedence_model.response(
        messages=[Message(role="user", content="precedence")]
    )
    (prec_span,) = span_exporter.get_finished_spans()
    assert (
        prec_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
        == 0.8
    )
    assert (
        prec_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TOP_P) == 0.75
    )
    assert (
        prec_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS)
        == 128
    )
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_K not in prec_span.attributes

    span_exporter.clear()

    # 4. request_params['config'] still replaces all lower-precedence config sources
    req_override_model = Gemini(
        id="gemini-2.5-flash",
        generation_config={"temperature": 0.3, "max_output_tokens": 128},
        generative_model_kwargs={"top_p": 0.75},
        temperature=0.8,
        request_params={"config": {"temperature": 0.15}},
        client=_make_client(),
    )
    req_override_model.response(
        messages=[Message(role="user", content="request_params override")]
    )
    (req_span,) = span_exporter.get_finished_spans()
    assert (
        req_span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE)
        == 0.15
    )
    assert GenAIAttributes.GEN_AI_REQUEST_TOP_P not in req_span.attributes
    assert GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS not in req_span.attributes


def test_openai_responses_input_file_url_content_block(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """OpenAI Responses input_file blocks with file_url are captured as document UriPart/BlobPart."""
    import base64

    import httpx
    import openai
    from agno.models.openai.responses import OpenAIResponses

    pdf_bytes = b"%PDF-1.4\nresponses-file"
    pdf_data_url = "data:application/pdf;base64," + base64.b64encode(
        pdf_bytes
    ).decode("ascii")

    payload = {
        "id": "resp_input_file_url",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Reviewed report.",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 16,
        },
    }
    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(200, json=payload)
            )
        ),
    )
    model = OpenAIResponses(id="gpt-4o-mini", client=client)
    model.response(
        messages=[
            Message(
                role="user",
                content=[
                    {
                        "type": "input_file",
                        "file_url": "https://example.com/report.pdf",
                    },
                    {
                        "type": "input_file",
                        "file_url": pdf_data_url,
                    },
                ],
            )
        ]
    )

    (span,) = span_exporter.get_finished_spans()
    input_msgs: Any = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert len(input_msgs) == 1
    assert list(input_msgs[0]["parts"]) == [
        {
            "type": "uri",
            "modality": "document",
            "mime_type": "application/pdf",
            "uri": "https://example.com/report.pdf",
        },
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": pdf_bytes,
        },
    ]


def test_model_endpoint_omitted_when_base_url_missing(
    instrument_agno,
    span_exporter,
    metric_reader,
) -> None:
    """When no base_url, host, or client endpoint is configured, neither server.address nor server.port is emitted."""

    class _NoEndpointModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="sync ok")

        async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(content="async ok")

    model = _NoEndpointModel(id="no-endpoint-model", provider="Custom")
    model.response(messages=[Message(role="user", content="sync no url")])
    asyncio.run(
        model.aresponse(
            messages=[Message(role="user", content="async no url")]
        )
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        assert "server.address" not in span.attributes
        assert "server.port" not in span.attributes

    metrics_data = metric_reader.get_metrics_data()
    observed_duration_points = 0
    for rm in metrics_data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == "gen_ai.client.operation.duration":
                    for dp in metric.data.data_points:
                        observed_duration_points += dp.count
                        assert "server.address" not in dp.attributes
                        assert "server.port" not in dp.attributes
    assert observed_duration_points == 2
