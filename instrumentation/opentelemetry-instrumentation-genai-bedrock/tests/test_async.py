# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import aiobotocore
import aiobotocore.session
import pytest
import pytest_asyncio
from aiobotocore.response import StreamingBody as AioStreamingBody
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock.patch import (
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

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
NOVA_MODEL_ID = "amazon.nova-micro-v1:0"


def _anthropic_body(text: str = "Hi", inp: int = 15, out: int = 7) -> bytes:
    return json.dumps(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": inp, "output_tokens": out},
        }
    ).encode("utf-8")


class _FakeHTTPContent:
    """Stands in for the aiohttp payload behind a real streaming body."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    async def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            chunk = self._data[self._pos :]
            self._pos = len(self._data)
            return chunk
        chunk = self._data[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk

    def at_eof(self) -> bool:
        return self._pos >= len(self._data)


class _FakeHTTPResponse:
    """Transport-level stand-in so tests exercise the SDK's real body class.

    Only the socket is faked: the object handed to the instrumentation is a
    genuine aiobotocore streaming body, so its read/iteration contract is the
    one under test.
    """

    def __init__(self, data: bytes) -> None:
        self.content = _FakeHTTPContent(data)
        self.url = "https://bedrock-runtime.us-east-1.amazonaws.com/"
        self.closed = False

    def at_eof(self) -> bool:
        return self.content.at_eof()

    def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> _FakeHTTPResponse:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.close()


def _streaming_body(data: bytes | None = None) -> AioStreamingBody:
    payload = _anthropic_body() if data is None else data
    return AioStreamingBody(_FakeHTTPResponse(payload), len(payload))


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


@pytest_asyncio.fixture
async def async_bedrock_client():
    """A real aiobotocore Bedrock runtime client."""
    session = aiobotocore.session.AioSession()
    async with session.create_client(
        "bedrock-runtime",
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    ) as client:
        yield client


def _stub(
    client: Any, operation: str, response: Any, **params: Any
) -> Stubber:
    stubber = Stubber(client)
    # Event streams and streaming bodies are not describable by the output
    # shape, so Stubber's response validation cannot accept them.
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        operation, service_response=response, expected_params=params
    )
    return stubber


CONVERSE_RESPONSE = {
    "output": {
        "message": {
            "role": "assistant",
            "content": [{"text": "Hello, async world!"}],
        }
    },
    "stopReason": "end_turn",
    "usage": {"inputTokens": 12, "outputTokens": 8},
}


@pytest.mark.asyncio
async def test_async_converse_success(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    with _stub(
        async_bedrock_client,
        "converse",
        CONVERSE_RESPONSE,
        modelId=NOVA_MODEL_ID,
        messages=messages,
    ):
        response = await async_bedrock_client.converse(
            modelId=NOVA_MODEL_ID, messages=messages
        )

    assert (
        response["output"]["message"]["content"][0]["text"]
        == "Hello, async world!"
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"chat {NOVA_MODEL_ID}"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_converse_client_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_bedrock_client)
    stubber.add_client_error(
        "converse",
        service_error_code="ValidationException",
        service_message="Invalid input",
        expected_params={"modelId": NOVA_MODEL_ID, "messages": []},
    )

    with stubber, pytest.raises(Exception):
        await async_bedrock_client.converse(modelId=NOVA_MODEL_ID, messages=[])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] in (
        "ValidationException",
        "botocore.errorfactory.ValidationException",
    )


@pytest.mark.asyncio
async def test_async_converse_records_cancelled_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    async def _cancel(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    async_bedrock_client._endpoint.make_request = _cancel

    with pytest.raises(asyncio.CancelledError):
        await async_bedrock_client.converse(modelId=NOVA_MODEL_ID, messages=[])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


CONVERSE_STREAM_EVENTS = [
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


@pytest.mark.asyncio
async def test_async_converse_stream_success(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    with _stub(
        async_bedrock_client,
        "converse_stream",
        {"stream": _MockAsyncEventStream(CONVERSE_STREAM_EVENTS)},
        modelId=NOVA_MODEL_ID,
        messages=[],
    ):
        response = await async_bedrock_client.converse_stream(
            modelId=NOVA_MODEL_ID, messages=[]
        )

        assert isinstance(
            response["stream"], AsyncBedrockConverseStreamWrapper
        )
        chunks = [chunk async for chunk in response["stream"]]
        assert len(chunks) == len(CONVERSE_STREAM_EVENTS)

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
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """A stream-side failure raised by the SDK mid-iteration."""
    events = [{"messageStart": {"role": "assistant"}}, {"fail": True}]
    with _stub(
        async_bedrock_client,
        "converse_stream",
        {"stream": _FailingAsyncEventStream(events)},
        modelId=NOVA_MODEL_ID,
        messages=[],
    ):
        response = await async_bedrock_client.converse_stream(
            modelId=NOVA_MODEL_ID, messages=[]
        )

        with pytest.raises(RuntimeError, match="Stream read failure"):
            async for _ in response["stream"]:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_async_converse_stream_caller_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """A caller-side failure raised inside `async with` before draining."""
    with _stub(
        async_bedrock_client,
        "converse_stream",
        {"stream": _MockAsyncEventStream(CONVERSE_STREAM_EVENTS)},
        modelId=NOVA_MODEL_ID,
        messages=[],
    ):
        response = await async_bedrock_client.converse_stream(
            modelId=NOVA_MODEL_ID, messages=[]
        )

        with pytest.raises(ValueError, match="caller error"):
            async with response["stream"] as stream:
                async for _ in stream:
                    raise ValueError("caller error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_async_invoke_model_records_cancelled_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    async def _cancel(*_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    async_bedrock_client._endpoint.make_request = _cancel

    with pytest.raises(asyncio.CancelledError):
        await async_bedrock_client.invoke_model(modelId=MODEL_ID, body="{}")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes[ErrorAttributes.ERROR_TYPE]
        == "asyncio.exceptions.CancelledError"
    )


@pytest.mark.asyncio
async def test_async_invoke_model_lazy_read(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body("Async invoke response")
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        # The span stays open until the caller drains the body.
        assert len(span_exporter.get_finished_spans()) == 0
        assert isinstance(response["body"], AsyncBedrockStreamingBodyWrapper)

        assert await response["body"].read() == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_invoke_model_body_forwards_attributes(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """Non-instrumented attributes must reach the wrapped body unchanged."""
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )
        body = response["body"]

        assert body.tell() == 0
        assert await body.read() == body_content
        assert body.tell() == len(body_content)


@pytest.mark.skipif(
    int(aiobotocore.__version__.split(".")[0]) < 3,
    reason=(
        "aiobotocore 2.x's StreamingBody is itself a wrapt proxy that reports "
        "the HTTP response as its __class__, so isinstance cannot survive a "
        "second proxy layer"
    ),
)
@pytest.mark.asyncio
async def test_async_invoke_model_body_is_transparent_proxy(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """isinstance must still resolve to the SDK's own body type."""
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {"contentType": "application/json", "body": _streaming_body()},
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )
        body = response["body"]

        assert isinstance(body, AioStreamingBody)
        await body.read()


@pytest.mark.asyncio
async def test_async_invoke_model_chunked_read(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body("Chunked", inp=5, out=3)
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
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
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body("Chunked then drain", inp=18, out=4)
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        first_chunk = await response["body"].read(20)
        assert len(first_chunk) == 20
        remainder = await response["body"].read()
        assert first_chunk + remainder == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 18
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4


@pytest.mark.asyncio
async def test_async_invoke_model_context_manager(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """`async with body as b` must bind the instrumented body, not the raw one."""
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        async with response["body"] as body:
            assert body is response["body"]
            assert await body.read() == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio
async def test_async_invoke_model_async_iteration(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        chunks = [chunk async for chunk in response["body"]]
        assert b"".join(chunks) == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15


@pytest.mark.asyncio
async def test_async_invoke_model_iter_chunks(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        chunks = [chunk async for chunk in response["body"].iter_chunks(16)]
        assert b"".join(chunks) == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15


@pytest.mark.asyncio
async def test_async_invoke_model_iter_lines(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        lines = [line async for line in response["body"].iter_lines()]
        assert b"".join(lines) == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15


@pytest.mark.asyncio
async def test_async_invoke_model_readinto(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    body_content = _anthropic_body()
    with _stub(
        async_bedrock_client,
        "invoke_model",
        {
            "contentType": "application/json",
            "body": _streaming_body(body_content),
        },
        modelId=MODEL_ID,
        body="{}",
    ):
        response = await async_bedrock_client.invoke_model(
            modelId=MODEL_ID, body="{}"
        )

        collected = bytearray()
        buffer = bytearray(32)
        while True:
            read = await response["body"].readinto(buffer)
            if read == 0:
                break
            collected.extend(buffer[:read])

        assert bytes(collected) == body_content

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15


INVOKE_MODEL_STREAM_EVENTS = [
    {
        "chunk": {
            "bytes": json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "role": "assistant",
                        "usage": {
                            "input_tokens": 20,
                            "cache_read_input_tokens": 5,
                        },
                    },
                }
            ).encode()
        }
    },
    {
        "chunk": {
            "bytes": json.dumps(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ).encode()
        }
    },
    {
        "chunk": {
            "bytes": json.dumps(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": "Async streaming invoke",
                    },
                }
            ).encode()
        }
    },
    {
        "chunk": {
            "bytes": json.dumps(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 8},
                }
            ).encode()
        }
    },
]


@pytest.mark.asyncio
async def test_async_invoke_model_stream_success(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    with _stub(
        async_bedrock_client,
        "invoke_model_with_response_stream",
        {"body": _MockAsyncEventStream(INVOKE_MODEL_STREAM_EVENTS)},
        modelId=MODEL_ID,
        body="{}",
    ):
        response = (
            await async_bedrock_client.invoke_model_with_response_stream(
                modelId=MODEL_ID, body="{}"
            )
        )

        assert isinstance(
            response["body"], AsyncBedrockInvokeModelStreamWrapper
        )
        chunks = [chunk async for chunk in response["body"]]
        assert len(chunks) == len(INVOKE_MODEL_STREAM_EVENTS)

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
async def test_async_invoke_model_stream_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """A stream-side failure raised by the SDK mid-iteration."""
    events = [
        {"chunk": {"bytes": b'{"type":"message_start"}'}},
        {"fail": True},
    ]
    with _stub(
        async_bedrock_client,
        "invoke_model_with_response_stream",
        {"body": _FailingAsyncEventStream(events)},
        modelId=MODEL_ID,
        body="{}",
    ):
        response = (
            await async_bedrock_client.invoke_model_with_response_stream(
                modelId=MODEL_ID, body="{}"
            )
        )

        with pytest.raises(RuntimeError, match="Stream read failure"):
            async for _ in response["body"]:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_async_invoke_model_stream_caller_error(
    async_bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    """A caller-side failure raised inside `async with` before draining."""
    with _stub(
        async_bedrock_client,
        "invoke_model_with_response_stream",
        {"body": _MockAsyncEventStream(INVOKE_MODEL_STREAM_EVENTS)},
        modelId=MODEL_ID,
        body="{}",
    ):
        response = (
            await async_bedrock_client.invoke_model_with_response_stream(
                modelId=MODEL_ID, body="{}"
            )
        )

        with pytest.raises(ValueError, match="caller error"):
            async with response["body"] as stream:
                async for _ in stream:
                    raise ValueError("caller error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_async_non_bedrock_service_is_not_instrumented(
    instrument_with_content,
    span_exporter,
) -> None:
    session = aiobotocore.session.AioSession()
    async with session.create_client(
        "s3", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    ) as s3_client:
        stubber = Stubber(s3_client)
        stubber.add_response("list_buckets", service_response={"Buckets": []})
        with stubber:
            await s3_client.list_buckets()

    assert span_exporter.get_finished_spans() == ()


def test_patch_and_unpatch_bedrock(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    patch_bedrock(handler)
    unpatch_bedrock()
