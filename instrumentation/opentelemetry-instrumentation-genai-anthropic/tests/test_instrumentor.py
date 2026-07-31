# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AnthropicInstrumentor class."""

from types import TracebackType
from typing import Any

import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import Message, TextBlock, Usage

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamManagerWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def _fake_message(model: str) -> Message:
    return Message(
        id="msg_test",
        content=[TextBlock(text="hello", type="text")],
        model=model,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=1, output_tokens=2),
    )


def _assert_chat_span(span_exporter, model: str) -> None:
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == f"chat {model}"
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == (
        GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_test"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == model
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 1
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 2


def test_instrumentor_instantiation():
    """Test that the instrumentor can be instantiated."""
    instrumentor = AnthropicInstrumentor()
    assert instrumentor is not None
    assert isinstance(instrumentor, AnthropicInstrumentor)


def test_instrumentation_dependencies():
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = AnthropicInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies is not None
    assert len(dependencies) > 0
    assert "anthropic >= 0.16.0" in dependencies


def test_instrument_uninstrument_cycle(
    tracer_provider, logger_provider, meter_provider
):
    """Test that instrument() and uninstrument() can be called multiple times."""
    instrumentor = AnthropicInstrumentor()

    # First instrumentation
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # First uninstrumentation
    instrumentor.uninstrument()

    # Second instrumentation (should work)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second uninstrumentation
    instrumentor.uninstrument()


def test_multiple_instrumentation_calls(
    tracer_provider, logger_provider, meter_provider
):
    """Test that multiple instrument() calls don't cause issues."""
    instrumentor = AnthropicInstrumentor()

    # First call
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Second call (should be idempotent or handle gracefully)
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_uninstrument_without_instrument():
    """Test that uninstrument() can be called without prior instrument()."""
    instrumentor = AnthropicInstrumentor()

    # This should not raise an error
    instrumentor.uninstrument()


def test_instrument_with_no_providers(
    tracer_provider, logger_provider, meter_provider
):
    """Test that instrument() works without explicit providers.

    Note: We still pass providers to ensure a clean test environment,
    but this tests that the instrumentor can be called and cleaned up.
    In a real scenario without explicit providers, it would use the
    global (no-op) providers.
    """
    instrumentor = AnthropicInstrumentor()

    # Test that instrument/uninstrument cycle works
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    # Clean up
    instrumentor.uninstrument()


def test_instrumentor_has_required_attributes():
    """Test that the instrumentor has the required methods."""
    instrumentor = AnthropicInstrumentor()

    assert hasattr(instrumentor, "instrument")
    assert hasattr(instrumentor, "uninstrument")
    assert hasattr(instrumentor, "instrumentation_dependencies")
    assert callable(instrumentor.instrument)
    assert callable(instrumentor.uninstrument)
    assert callable(instrumentor.instrumentation_dependencies)


def test_messages_parse_is_instrumented(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    """Messages.parse should emit the same chat telemetry as Messages.create."""
    model = "claude-test"

    def fake_parse(self: object, **kwargs: Any) -> Message:
        return _fake_message(kwargs["model"])

    # parse() lands on both the sync and async resource classes together in the
    # real SDK, so add it to both to match the shape the instrumentor wraps.
    monkeypatch.setattr(Messages, "parse", fake_parse, raising=False)
    monkeypatch.setattr(AsyncMessages, "parse", fake_parse, raising=False)

    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        response = Anthropic().messages.parse(
            model=model,
            max_tokens=10,
            messages=[{"role": "user", "content": "hello"}],
            output_format={"type": "json_schema", "schema": {}},
        )

    assert response.id == "msg_test"
    _assert_chat_span(span_exporter, model)


@pytest.mark.asyncio
async def test_async_messages_parse_is_instrumented(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    """AsyncMessages.parse should emit the same chat telemetry as create."""
    model = "claude-test"

    async def fake_parse(self: object, **kwargs: Any) -> Message:
        return _fake_message(kwargs["model"])

    def fake_sync_parse(self: object, **kwargs: Any) -> Message:
        return _fake_message(kwargs["model"])

    # parse() lands on both the sync and async resource classes together in the
    # real SDK, so add it to both to match the shape the instrumentor wraps.
    monkeypatch.setattr(AsyncMessages, "parse", fake_parse, raising=False)
    monkeypatch.setattr(Messages, "parse", fake_sync_parse, raising=False)

    async with AsyncAnthropic() as client:
        with instrument(
            AnthropicInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            response = await client.messages.parse(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
                output_format={"type": "json_schema", "schema": {}},
            )

    assert response.id == "msg_test"
    _assert_chat_span(span_exporter, model)


class _FakeAsyncResponse:
    async def aclose(self) -> None:
        return None


class _FakeAsyncStream:
    def __init__(self, message: Message):
        self.current_message_snapshot = message
        self.response = _FakeAsyncResponse()
        self._chunks = iter([object()])

    def __aiter__(self) -> "_FakeAsyncStream":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def close(self) -> None:
        return None


class _FakeAsyncStreamManager:
    def __init__(self, message: Message):
        self._message = message

    async def __aenter__(self) -> _FakeAsyncStream:
        return _FakeAsyncStream(self._message)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return False


@pytest.mark.asyncio
async def test_async_messages_stream_is_instrumented(
    monkeypatch,
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
):
    """AsyncMessages.stream should emit telemetry when the stream is consumed."""
    model = "claude-test"

    def fake_stream(self: object, **kwargs: Any) -> _FakeAsyncStreamManager:
        return _FakeAsyncStreamManager(_fake_message(kwargs["model"]))

    monkeypatch.setattr(AsyncMessages, "stream", fake_stream)

    async with AsyncAnthropic() as client:
        with instrument(
            AnthropicInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
        ):
            manager = client.messages.stream(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "hello"}],
            )
            assert isinstance(manager, AsyncMessagesStreamManagerWrapper)
            async with manager as stream:
                async for _ in stream:
                    pass

    _assert_chat_span(span_exporter, model)
