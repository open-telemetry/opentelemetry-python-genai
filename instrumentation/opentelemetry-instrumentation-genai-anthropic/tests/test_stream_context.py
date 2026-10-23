# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streams must not keep the chat span current while they are unconsumed."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from anthropic.types import (
    Message,
    RawMessageStartEvent,
    Usage,
)

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamWrapper,
    MessagesStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler

_MODEL = "claude-sonnet-4-20250514"
_MESSAGES = [{"role": "user", "content": "Say hello in one word."}]


def _start_event() -> RawMessageStartEvent:
    return RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_1",
            type="message",
            role="assistant",
            model=_MODEL,
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=1, output_tokens=0),
        ),
    )


def _assert_no_detach_errors(caplog) -> None:
    assert not [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= 40
    ]


def _assert_caller_parents_both(span_exporter, caller_span_id: int) -> None:
    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    chat = next(s for n, s in spans.items() if n.startswith("chat"))
    middle = spans["middle"]
    assert chat.parent is not None
    assert chat.parent.span_id == caller_span_id
    assert middle.parent is not None
    assert middle.parent.span_id == caller_span_id


def _is_current(caller: trace.Span) -> bool:
    return trace.get_current_span() is caller


# --- real SDK paths, replayed from cassettes ---------------------------------


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_create_streaming")
def test_sync_create_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        stream = anthropic_client.messages.create(
            model=_MODEL, max_tokens=100, messages=_MESSAGES, stream=True
        )
        assert _is_current(caller)
        with tracer.start_as_current_span("middle"):
            pass
        assert _is_current(caller)
        for _ in stream:
            assert _is_current(caller)
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_stream")
def test_sync_messages_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with anthropic_client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            assert _is_current(caller)
            with tracer.start_as_current_span("middle"):
                pass
            for _ in stream:
                assert _is_current(caller)
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_create_streaming_with_raw_response")
def test_sync_raw_response_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        raw_response = anthropic_client.messages.with_raw_response.create(
            model=_MODEL, max_tokens=100, messages=_MESSAGES, stream=True
        )
        assert _is_current(caller)
        with tracer.start_as_current_span("middle"):
            pass
        stream = raw_response.parse()
        assert _is_current(caller)
        for _ in stream:
            assert _is_current(caller)
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_create_streaming")
async def test_async_create_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        stream = await async_anthropic_client.messages.create(
            model=_MODEL, max_tokens=100, messages=_MESSAGES, stream=True
        )
        assert _is_current(caller)
        with tracer.start_as_current_span("middle"):
            pass

        async def drain() -> None:
            async for _ in stream:
                pass

        await asyncio.create_task(drain())
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_stream")
async def test_async_messages_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        async with async_anthropic_client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            assert _is_current(caller)
            with tracer.start_as_current_span("middle"):
                pass
            async for _ in stream:
                assert _is_current(caller)
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_create_streaming_with_raw_response")
async def test_async_raw_response_stream_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        raw_response = (
            await async_anthropic_client.messages.with_raw_response.create(
                model=_MODEL, max_tokens=100, messages=_MESSAGES, stream=True
            )
        )
        assert _is_current(caller)
        with tracer.start_as_current_span("middle"):
            pass
        # Older SDKs return the stream from a synchronous parse().
        stream = raw_response.parse()
        if inspect.isawaitable(stream):
            stream = await stream

        async def drain() -> None:
            async for _ in stream:
                pass

        await asyncio.create_task(drain())
        assert _is_current(caller)
    _assert_caller_parents_both(
        span_exporter, caller.get_span_context().span_id
    )
    _assert_no_detach_errors(caplog)


# --- wrapper level -----------------------------------------------------------


def _assert_invocation_current(invocation: Any) -> None:
    assert (
        trace.get_current_span().get_span_context()
        == trace.get_current_span(invocation.context).get_span_context()
    )


def _events(
    invocation: Any, event: Any, error: BaseException | None
) -> Iterator[Any]:
    _assert_invocation_current(invocation)
    yield event
    _assert_invocation_current(invocation)
    if error is not None:
        raise error


async def _aevents(
    invocation: Any, event: Any, error: BaseException | None
) -> AsyncIterator[Any]:
    for item in _events(invocation, event, error):
        await asyncio.sleep(0)
        yield item


def _invocation(tracer_provider, logger_provider, meter_provider) -> Any:
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        instrumentation_scope_name=(
            "opentelemetry.instrumentation.genai.anthropic"
        ),
    ).inference("anthropic", request_model=_MODEL)


@pytest.mark.parametrize("ending", ["drain", "close", "error"])
def test_sync_wrapper_activates_only_during_reads(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    ending: str,
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    error = ConnectionError("stream failed") if ending == "error" else None
    event = _start_event()
    stream = MessagesStreamWrapper(
        _events(invocation, event, error), invocation, False
    )
    assert context.get_current() is before
    assert next(stream) is event
    assert context.get_current() is before
    assert span_exporter.get_finished_spans() == ()

    if ending == "close":
        stream.close()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            next(stream)
        assert raised.value is error
    else:
        assert list(stream) == []
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1


def test_sync_wrapper_close_before_read_keeps_context(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(
        _events(invocation, _start_event(), None),
        invocation,
        False,
    )
    assert context.get_current() is before
    stream.close()
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["drain", "close", "error"])
async def test_async_wrapper_activates_only_during_reads(
    tracer_provider,
    logger_provider,
    meter_provider,
    span_exporter,
    ending: str,
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    error = ConnectionError("stream failed") if ending == "error" else None
    event = _start_event()
    stream = AsyncMessagesStreamWrapper(
        _aevents(invocation, event, error), invocation, False
    )
    assert context.get_current() is before
    assert await asyncio.create_task(anext(stream)) is event
    assert context.get_current() is before
    assert span_exporter.get_finished_spans() == ()

    async def drain() -> list[Any]:
        return [item async for item in stream]

    if ending == "close":
        await stream.aclose()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            await asyncio.create_task(anext(stream))
        assert raised.value is error
    else:
        assert await asyncio.create_task(drain()) == []
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_wrapper_close_before_read_keeps_context(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = AsyncMessagesStreamWrapper(
        _aevents(invocation, _start_event(), None),
        invocation,
        False,
    )
    assert context.get_current() is before
    await stream.aclose()
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
