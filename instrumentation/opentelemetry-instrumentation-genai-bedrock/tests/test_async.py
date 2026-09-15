# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_async_converse,
    _handle_async_invoke_model,
    _make_aio_api_call_wrapper,
    patch_bedrock,
    unpatch_bedrock,
)
from opentelemetry.instrumentation.genai.bedrock.stream import (
    AsyncBedrockConverseStreamWrapper,
    AsyncBedrockInvokeModelStreamWrapper,
    AsyncBedrockStreamingBodyWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _bedrock_client() -> Any:
    return SimpleNamespace(
        meta=SimpleNamespace(
            endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com"
        ),
        _service_model=SimpleNamespace(service_name="bedrock-runtime"),
    )


class _MockAsyncStreamingBody:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self.closed = False

    async def read(self, amt: int | None = None) -> bytes:
        if self._pos >= len(self._data):
            return b""
        if amt is None:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + amt]
        self._pos += len(chunk)
        return chunk

    async def aclose(self) -> None:
        self.closed = True

    def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _MockAsyncStreamingBody:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()


class _MockAsyncEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> _MockAsyncEventStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FailingAsyncEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> _FailingAsyncEventStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            val = next(self._iter)
            if "fail" in val:
                raise RuntimeError("Stream read failure")
            return val
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_async_converse_records_cancelled_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def _cancelled_call(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _handle_async_converse(
            _cancelled_call,
            _bedrock_client(),
            (),
            {},
            {
                "modelId": "amazon.nova-micro-v1:0",
                "messages": [],
            },
            handler,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


@pytest.mark.asyncio
async def test_async_converse_success(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def _success_call(*_args: Any, **_kwargs: Any) -> Any:
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello, async world!"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 12,
                "outputTokens": 8,
            },
        }

    response = await _handle_async_converse(
        _success_call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": "Hello"}],
                }
            ],
        },
        handler,
    )
    assert (
        response["output"]["message"]["content"][0]["text"]
        == "Hello, async world!"
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_converse_stream_success(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "Async "},
            }
        },
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "stream!"},
            }
        },
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 2,
                    "cacheWriteInputTokens": 1,
                }
            }
        },
    ]

    async def _stream_call(*_args: Any, **_kwargs: Any) -> Any:
        return {"stream": _MockAsyncEventStream(events)}

    response = await _handle_async_converse(
        _stream_call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [],
        },
        handler,
        is_stream=True,
    )

    assert isinstance(response["stream"], AsyncBedrockConverseStreamWrapper)
    chunks = []
    async for chunk in response["stream"]:
        chunks.append(chunk)
    assert len(chunks) == len(events)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 2
    )
    assert span.attributes["gen_ai.usage.cache_write.input_tokens"] == 1
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_converse_stream_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    events = [
        {"messageStart": {"role": "assistant"}},
        {"fail": True},
    ]

    async def _stream_call(*_args: Any, **_kwargs: Any) -> Any:
        return {"stream": _FailingAsyncEventStream(events)}

    response = await _handle_async_converse(
        _stream_call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "amazon.nova-micro-v1:0",
            "messages": [],
        },
        handler,
        is_stream=True,
    )

    with pytest.raises(RuntimeError, match="Stream read failure"):
        async for _ in response["stream"]:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_async_invoke_model_records_cancelled_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def _cancelled_call(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _handle_async_invoke_model(
            _cancelled_call,
            _bedrock_client(),
            (),
            {},
            {
                "modelId": "anthropic.claude-v2",
                "body": "{}",
            },
            handler,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


@pytest.mark.asyncio
async def test_async_invoke_model_lazy_read(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    body_content = b'{"type":"message","role":"assistant","content":[{"type":"text","text":"Async invoke response"}],"stop_reason":"end_turn","usage":{"input_tokens":15,"output_tokens":7}}'
    mock_body = _MockAsyncStreamingBody(body_content)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {"body": mock_body}

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": '{"messages":[{"role":"user","content":"Hi"}]}',
        },
        handler,
    )

    # Before read() is called, the span should NOT be finished yet
    assert len(span_exporter.get_finished_spans()) == 0
    assert isinstance(response["body"], AsyncBedrockStreamingBodyWrapper)

    # Now read the body
    data = await response["body"].read()
    assert data == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_invoke_model_chunked_read(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    body_content = b'{"type":"message","role":"assistant","content":[{"type":"text","text":"Chunked"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}'
    mock_body = _MockAsyncStreamingBody(body_content)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {"body": mock_body}

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": '{"messages":[]}',
        },
        handler,
    )

    chunks = []
    while True:
        chunk = await response["body"].read(20)
        if not chunk:
            break
        chunks.append(chunk)

    assert b"".join(chunks) == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5


@pytest.mark.asyncio
async def test_async_invoke_model_chunked_then_drain_read(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    body_content = b'{"type":"message","role":"assistant","content":[{"type":"text","text":"Chunked then drain"}],"stop_reason":"end_turn","usage":{"input_tokens":18,"output_tokens":4}}'
    mock_body = _MockAsyncStreamingBody(body_content)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {"body": mock_body}

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": '{"messages":[]}',
        },
        handler,
    )

    # Read the first 20 bytes as a chunk
    first_chunk = await response["body"].read(20)
    assert len(first_chunk) == 20

    # Drain the remainder with read() (amt=None)
    remainder = await response["body"].read()
    assert first_chunk + remainder == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 18
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4


@pytest.mark.asyncio
async def test_async_invoke_model_stream_success(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    events = [
        {
            "chunk": {
                "bytes": b'{"type":"message_start","message":{"role":"assistant","usage":{"input_tokens":20,"cache_read_input_tokens":5}}}'
            }
        },
        {
            "chunk": {
                "bytes": b'{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}'
            }
        },
        {
            "chunk": {
                "bytes": b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Async streaming invoke"}}'
            }
        },
        {
            "chunk": {
                "bytes": b'{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}'
            }
        },
    ]

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {"body": _MockAsyncEventStream(events)}

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": "{}",
        },
        handler,
        is_stream=True,
    )

    assert isinstance(response["body"], AsyncBedrockInvokeModelStreamWrapper)
    chunks = []
    async for chunk in response["body"]:
        chunks.append(chunk)
    assert len(chunks) == len(events)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 5
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_make_aio_api_call_wrapper_routing(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    wrapper = _make_aio_api_call_wrapper(handler)

    client = _bedrock_client()

    async def original_func(op: str, params: dict[str, Any]) -> Any:
        if op == "Converse":
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
            }
        elif op == "NonBedrockOp":
            return {"result": "ok"}
        return {}

    wrapped_fn = wrapper(
        original_func, client, ("Converse", {"modelId": "m"}), {}
    )
    res = await wrapped_fn

    assert "output" in res

    # Non-bedrock service
    non_bedrock_client = SimpleNamespace(
        meta=SimpleNamespace(endpoint_url="https://s3.amazonaws.com"),
        _service_model=SimpleNamespace(service_name="s3"),
    )
    wrapped_s3 = wrapper(
        original_func,
        non_bedrock_client,
        ("GetObject", {}),
        {},
    )
    res_s3 = await wrapped_s3
    assert res_s3 == {}

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1


def test_patch_and_unpatch_bedrock(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    patch_bedrock(handler)
    unpatch_bedrock()


@pytest.mark.asyncio
async def test_async_invoke_model_titan_embeddings(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    body_content = b'{"embedding":[0.1,0.2,0.3,0.4],"inputTextTokenCount":7}'
    mock_body = _MockAsyncStreamingBody(body_content)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {
            "body": mock_body,
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "7",
                }
            },
        }

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "amazon.titan-embed-text-v1",
            "body": '{"inputText":"This is the text to embed."}',
        },
        handler,
    )

    # Before read() is called, the span should NOT be finished yet
    assert len(span_exporter.get_finished_spans()) == 0
    assert isinstance(response["body"], AsyncBedrockStreamingBodyWrapper)

    # Now read the body
    data = await response["body"].read()
    assert data == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "amazon.titan-embed-text-v1"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 4
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7


@pytest.mark.asyncio
async def test_async_invoke_model_titan_embeddings_chunked(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    body_content = b'{"embedding":[0.5,0.6,0.7],"inputTextTokenCount":5}'
    mock_body = _MockAsyncStreamingBody(body_content)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        return {"body": mock_body}

    response = await _handle_async_invoke_model(
        _call,
        _bedrock_client(),
        (),
        {},
        {
            "modelId": "amazon.titan-embed-text-v1",
            "body": '{"inputText":"Test chunked embedding"}',
        },
        handler,
    )

    # Read in chunks
    chunk1 = await response["body"].read(10)
    assert len(span_exporter.get_finished_spans()) == 0
    chunk2 = await response["body"].read(None)
    assert chunk1 + chunk2 == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 3
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5


@pytest.mark.asyncio
async def test_async_invoke_model_embedding_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)

    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Embedding failure")

    with pytest.raises(RuntimeError, match="Embedding failure"):
        await _handle_async_invoke_model(
            _call,
            _bedrock_client(),
            (),
            {},
            {
                "modelId": "amazon.titan-embed-text-v1",
                "body": '{"inputText":"Test failure"}',
            },
            handler,
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
