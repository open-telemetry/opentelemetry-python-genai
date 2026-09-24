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
    assert output_messages[0]["finish_reason"] == "stop"


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
    """Test that closing a sync response stream before it is drained finalizes the span once."""

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
                    response_usage=MessageMetrics(
                        input_tokens=15, output_tokens=8
                    ),
                )
            return ModelResponse(
                content="The result is 42.",
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

    # Verify finish reasons: first turn is tool_calls, second turn is stop
    assert chat_spans[0].attributes.get(
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
    ) == ("tool_calls",)
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
