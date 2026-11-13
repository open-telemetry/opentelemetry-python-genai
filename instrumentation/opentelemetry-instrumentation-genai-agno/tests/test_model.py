# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Model inference instrumentation."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agno.agent import Agent
from agno.models.base import MessageData
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

    input_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_messages_raw is not None
    input_messages = json.loads(input_messages_raw)
    assert len(input_messages) == 2
    assert input_messages[0]["role"] == "system"
    assert input_messages[0]["parts"][0]["content"] == "System prompt"
    assert input_messages[1]["role"] == "user"
    assert input_messages[1]["parts"][0]["content"] == "User prompt"

    output_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
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

    input_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_messages_raw is not None
    input_messages = json.loads(input_messages_raw)
    assert len(input_messages) == 1
    assert input_messages[0]["parts"][0]["content"] == "Async prompt"

    output_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
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

    output_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
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

    output_messages_raw = span.attributes.get(
        GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert output_messages_raw is not None
    output_messages = json.loads(output_messages_raw)
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
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("error",)


def test_model_stream_error_caller_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller-side errors inside context manager mark the span with ERROR status."""

    class InfiniteStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            while True:
                yield ModelResponse(content="endless")

    model = InfiniteStreamModel(id="caller-err-model", provider="OpenAI")

    with pytest.raises(RuntimeError, match="Caller aborted"):
        with model.process_response_stream(
            messages=[Message(role="user", content="Hi")],
            assistant_message=Message(role="assistant"),
            stream_data=MessageData(),
        ) as stream:
            for _ in stream:
                raise RuntimeError("Caller aborted")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"


def test_model_response_stream_early_close(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that closing a sync response stream before it is drained finalizes the span once with interrupted finish reason."""

    class MultiChunkStreamModel(MockModel):
        def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="chunk 1")
            yield ModelResponse(content="chunk 2")
            yield ModelResponse(content="chunk 3")

    model = MultiChunkStreamModel(id="early-close-model", provider="OpenAI")
    stream = model.process_response_stream(
        messages=[Message(role="user", content="Hi")],
        assistant_message=Message(role="assistant"),
        stream_data=MessageData(),
    )
    first = next(stream)
    assert first.content == "chunk 1"
    stream.close()

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

    stream = _LengthStream(id="m", provider="OpenAI").process_response_stream(
        messages=[Message(role="user", content="hi")],
        assistant_message=Message(role="assistant"),
        stream_data=MessageData(),
    )
    next(stream)
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
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("error",)


def test_model_aresponse_stream_error_caller_side(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that caller-side errors inside async context manager mark the span with ERROR status."""

    class AsyncInfiniteStreamModel(MockModel):
        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            while True:
                yield ModelResponse(content="endless")

    model = AsyncInfiniteStreamModel(
        id="async-caller-err-model", provider="Anthropic"
    )

    async def _run() -> None:
        async with model.aprocess_response_stream(
            messages=[Message(role="user", content="Hi")],
            assistant_message=Message(role="assistant"),
            stream_data=MessageData(),
        ) as stream:
            async for _ in stream:
                raise RuntimeError("Async caller aborted")

    with pytest.raises(RuntimeError, match="Async caller aborted"):
        asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ERROR_TYPE) == "RuntimeError"


def test_model_aresponse_stream_early_close(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that closing an async response stream early finalizes the span once."""

    class AsyncMultiChunkStreamModel(MockModel):
        async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
            yield ModelResponse(content="async chunk 1")
            yield ModelResponse(content="async chunk 2")

    model = AsyncMultiChunkStreamModel(
        id="async-early-close-model", provider="Anthropic"
    )

    async def _run() -> None:
        stream = model.aprocess_response_stream(
            messages=[Message(role="user", content="Hi")],
            assistant_message=Message(role="assistant"),
            stream_data=MessageData(),
        )
        first = await stream.__anext__()
        assert first.content == "async chunk 1"
        await stream.aclose()

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

    class ToolArgsModel(MockModel):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                content=None,
                tool_calls=[
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
                response_usage=MessageMetrics(
                    input_tokens=10, output_tokens=5
                ),
            )

    model = ToolArgsModel(id="tool-args-model", provider="OpenAI")
    model._process_model_response(
        messages=[
            Message(
                role="assistant",
                tool_calls=[
                    {
                        "id": "hist_call",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query": "otel"}',
                        },
                    }
                ],
            )
        ],
        assistant_message=Message(role="assistant"),
        model_response=ModelResponse(),
    )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    input_msgs = json.loads(
        str(span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES))
    )
    assert input_msgs[0]["parts"][0]["arguments"] == {"query": "otel"}

    output_msgs = json.loads(
        str(span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
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

    input_msgs = json.loads(
        str(span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES))
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
        "content": b64encode(raw_image).decode(),
    }
    assert parts[3] == {
        "type": "blob",
        "modality": "audio",
        "mime_type": "audio/wav",
        "content": b64encode(raw_audio).decode(),
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

    output_msgs = json.loads(
        str(span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
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
        "content": b64encode(raw_audio).decode(),
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
    output_msgs = json.loads(
        str(second.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES))
    )
    assert output_msgs[0]["parts"] == [{"type": "text", "content": "done"}]


def test_gemini_reasoning_tokens_added_to_output_tokens_and_max_output_tokens(
    instrument_agno,
    span_exporter,
) -> None:
    """Test Gemini adds reasoning_tokens to output_tokens and captures max_output_tokens and audio tokens."""

    class _GeminiLikeModel(MockModel):
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

    model = _GeminiLikeModel(id="gemini-2.5-flash", provider="Google")
    model.response(messages=[Message(role="user", content="Add 123 and 456")])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "gcp.gemini"
    assert attrs.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 512
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS) == 31
    assert attrs.get(GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS) == 74
    assert (
        attrs.get(GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS) == 54
    )
    assert attrs.get("gen_ai.usage.audio.input_tokens") == 5
    assert attrs.get("gen_ai.usage.audio.output_tokens") == 10


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
    model = _SyncModel(id="gpt-4o", provider="OpenAI")

    with instrument(
        AgnoInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ):
        model._process_model_response(
            messages=[Message(role="user", content="hi")],
            assistant_message=Message(role="assistant"),
            model_response=ModelResponse(),
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
        model._process_model_response(
            messages=[Message(role="user", content="hi")],
            assistant_message=Message(role="assistant"),
            model_response=ModelResponse(),
            tools=tool_schema,
        )

    assert len(calls) == 1
    (capture_span,) = span_exporter.get_finished_spans()
    tool_defs = json.loads(
        str(capture_span.attributes[GenAIAttributes.GEN_AI_TOOL_DEFINITIONS])
    )
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
    stream = sync_model.process_response_stream(
        messages=[Message(role="user", content="gen image")],
        assistant_message=Message(role="assistant"),
        stream_data=MessageData(),
    )
    next(stream)
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
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
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert (
        out_msgs[0]["parts"][0]["uri"] == "https://example.com/stream-img.png"
    )

    # 3. Async early close
    span_exporter.clear()

    async def _async_early_close() -> None:
        astream = sync_model.aprocess_response_stream(
            messages=[Message(role="user", content="gen image")],
            assistant_message=Message(role="assistant"),
            stream_data=MessageData(),
        )
        await astream.__anext__()
        await astream.aclose()

    asyncio.run(_async_early_close())
    (span,) = span_exporter.get_finished_spans()
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
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
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
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
    input_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    )
    assert len(input_msgs) == 2

    first_parts = input_msgs[0]["parts"]
    assert first_parts[0] == {
        "type": "tool_call_response",
        "id": "call_struct",
        "response": [
            {"item": "apple", "count": 3},
            {"item": "pear", "count": 5},
        ],
    }
    part_types = [p["type"] for p in first_parts[1:]]
    assert part_types == ["uri", "blob", "uri", "uri"]

    second_parts = input_msgs[1]["parts"]
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
    input_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    )
    parts = input_msgs[0]["parts"]
    assert parts == [
        {"type": "text", "content": "Describe this"},
        {
            "type": "uri",
            "modality": "image",
            "mime_type": None,
            "uri": "https://example.com/cat.png",
        },
        {
            "type": "blob",
            "modality": "image",
            "mime_type": "image/png",
            "content": png_b64,
        },
        {
            "type": "blob",
            "modality": "audio",
            "mime_type": "audio/wav",
            "content": wav_b64,
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
    txt_b64 = base64.b64encode(b"Plain text doc").decode("ascii")

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
    input_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    )
    assert [m["role"] for m in input_msgs] == [
        "developer",
        "assistant",
        "user",
    ]
    assert input_msgs[2]["parts"] == [
        {
            "type": "blob",
            "modality": "document",
            "mime_type": "application/pdf",
            "content": pdf_b64,
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
            "content": txt_b64,
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
    out_msgs = json.loads(
        str(sync_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
    out_msgs = json.loads(
        str(sync_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
    stream = model.process_response_stream(
        messages=[Message(role="user", content="Weather?")],
        assistant_message=Message(role="assistant"),
        stream_data=MessageData(),
    )
    next(stream)
    next(stream)
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
        astream = model.aprocess_response_stream(
            messages=[Message(role="user", content="Weather?")],
            assistant_message=Message(role="assistant"),
            stream_data=MessageData(),
        )
        await astream.__anext__()
        await astream.aclose()

    asyncio.run(_async_close())
    (span,) = span_exporter.get_finished_spans()
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
    out_msgs = json.loads(
        str(span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    assert out_msgs[0]["parts"] == [
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
        max_tokens=0,
        max_completion_tokens=100,
        request_params={
            "temperature": 0,
            "frequency_penalty": 0.25,
            "stop": ["STOP_HERE"],
        },
        client=client,
    )
    model.response(messages=[Message(role="user", content="Hi")])

    assert sent_bodies[0]["temperature"] == 0
    assert sent_bodies[0]["max_tokens"] == 0
    assert sent_bodies[0]["frequency_penalty"] == 0.25
    assert sent_bodies[0]["stop"] == ["STOP_HERE"]

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE) == 0.0
    )
    assert span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS) == 0
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY)
        == 0.25
    )
    assert span.attributes.get(
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES
    ) == ("STOP_HERE",)


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
    sync_out = json.loads(
        str(sync_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
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
    async_out = json.loads(
        str(async_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
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
    sync_out = json.loads(
        str(sync_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    audio_blobs = [
        p
        for p in sync_out[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(audio_blobs) == 1
    assert audio_blobs[0]["content"] == "AAE="

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
    stream_out = json.loads(
        str(stream_span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
    )
    stream_audio_blobs = [
        p
        for p in stream_out[0]["parts"]
        if p.get("type") == "blob" and p.get("modality") == "audio"
    ]
    assert len(stream_audio_blobs) == 1
    assert stream_audio_blobs[0]["content"] == "AAE="
