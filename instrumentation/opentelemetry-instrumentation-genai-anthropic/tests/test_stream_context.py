# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streams must not keep the chat span current while they are unconsumed."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib
from anthropic.types import (
    Message,
    RawMessageStartEvent,
    Usage,
)

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.anthropic._raw_response import (
    wrap_raw_response,
)
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


# --- raw response, body read directly ----------------------------------------

_BODY = [b"one\n", b"two\n", b"three\n"]
_REQUEST = _http_lib.Request("POST", "https://api.anthropic.com/v1/messages")


class _ObservedStream(_http_lib.SyncByteStream):
    """Records the current span each time the response body is touched."""

    def __init__(self, seen: list[trace.SpanContext]) -> None:
        self._seen = seen

    def __iter__(self) -> Iterator[bytes]:
        for chunk in _BODY:
            self._seen.append(trace.get_current_span().get_span_context())
            yield chunk

    def close(self) -> None:
        self._seen.append(trace.get_current_span().get_span_context())


class _ObservedAsyncStream(_http_lib.AsyncByteStream):
    def __init__(self, seen: list[trace.SpanContext]) -> None:
        self._seen = seen

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in _BODY:
            await asyncio.sleep(0)
            self._seen.append(trace.get_current_span().get_span_context())
            yield chunk

    async def aclose(self) -> None:
        self._seen.append(trace.get_current_span().get_span_context())


def _body_proxy(invocation: Any, stream: Any) -> Any:
    response = _http_lib.Response(200, stream=stream, request=_REQUEST)
    return wrap_raw_response(
        SimpleNamespace(http_response=response), invocation, False
    )


def _assert_all_in_invocation(
    seen: list[trace.SpanContext], invocation: Any
) -> None:
    expected = trace.get_current_span(invocation.context).get_span_context()
    assert seen
    assert all(span_context == expected for span_context in seen)


@pytest.mark.parametrize(
    "method", ["iter_raw", "iter_bytes", "iter_text", "iter_lines"]
)
def test_sync_raw_body_iteration_activates_per_chunk(
    tracer_provider, logger_provider, meter_provider, span_exporter, method
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedStream(seen))

    chunks = getattr(proxy.http_response, method)()
    assert context.get_current() is before
    for _ in chunks:
        assert context.get_current() is before
        assert span_exporter.get_finished_spans() == ()
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.parametrize("method", ["iter_raw", "iter_bytes"])
def test_sync_raw_body_iterator_closed_early(
    tracer_provider, logger_provider, meter_provider, span_exporter, method
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedStream(seen))

    chunks = getattr(proxy.http_response, method)()
    next(chunks)
    assert context.get_current() is before
    chunks.close()
    proxy.http_response.close()
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


def test_sync_raw_body_read_activates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedStream(seen))

    assert proxy.http_response.read() == b"".join(_BODY)
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


def test_sync_raw_body_close_without_read_activates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedStream(seen))

    proxy.http_response.close()
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["aiter_raw", "aiter_bytes", "aiter_text", "aiter_lines"]
)
async def test_async_raw_body_iteration_activates_per_chunk(
    tracer_provider, logger_provider, meter_provider, span_exporter, method
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedAsyncStream(seen))

    async def drain() -> None:
        async for _ in getattr(proxy.http_response, method)():
            assert context.get_current() is before
            assert span_exporter.get_finished_spans() == ()

    await asyncio.create_task(drain())
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["aiter_raw", "aiter_bytes"])
async def test_async_raw_body_iterator_closed_early(
    tracer_provider, logger_provider, meter_provider, span_exporter, method
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedAsyncStream(seen))

    chunks = getattr(proxy.http_response, method)()
    await asyncio.create_task(anext(chunks))
    assert context.get_current() is before
    await chunks.aclose()
    await proxy.http_response.aclose()
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_raw_body_read_activates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedAsyncStream(seen))

    assert await proxy.http_response.aread() == b"".join(_BODY)
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_raw_body_close_without_read_activates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedAsyncStream(seen))

    await proxy.http_response.aclose()
    assert context.get_current() is before
    _assert_all_in_invocation(seen, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


# --- raw response, caller error ----------------------------------------------


def _record_current_span_on_finish(
    invocation: Any, seen: list[trace.SpanContext]
) -> None:
    apply_finish = invocation._apply_finish

    def spy(error: Any = None) -> None:
        seen.append(trace.get_current_span().get_span_context())
        apply_finish(error)

    invocation._apply_finish = spy


def test_sync_raw_response_caller_error_fails_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    finish_seen: list[trace.SpanContext] = []
    _record_current_span_on_finish(invocation, finish_seen)
    proxy = _body_proxy(invocation, _ObservedStream([]))

    proxy._fail(RuntimeError("caller failed"))

    assert context.get_current() is before
    _assert_all_in_invocation(finish_seen, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_raw_response_caller_error_fails_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    finish_seen: list[trace.SpanContext] = []
    _record_current_span_on_finish(invocation, finish_seen)
    proxy = _body_proxy(invocation, _ObservedAsyncStream([]))

    proxy._fail(RuntimeError("caller failed"))
    await proxy.http_response.aclose()

    assert context.get_current() is before
    _assert_all_in_invocation(finish_seen, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR


def test_sync_raw_response_caller_error_after_parse_fails_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    finish_seen: list[trace.SpanContext] = []
    _record_current_span_on_finish(invocation, finish_seen)
    proxy = _body_proxy(invocation, _ObservedStream([]))
    proxy._self_span_open = False
    proxy._self_stream_wrapper = MessagesStreamWrapper(
        iter(()), invocation, False
    )

    proxy._fail(RuntimeError("caller failed"))

    assert context.get_current() is before
    _assert_all_in_invocation(finish_seen, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_raw_response_caller_error_after_parse_fails_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    finish_seen: list[trace.SpanContext] = []
    _record_current_span_on_finish(invocation, finish_seen)
    proxy = _body_proxy(invocation, _ObservedAsyncStream([]))
    proxy._self_span_open = False
    proxy._self_stream_wrapper = AsyncMessagesStreamWrapper(
        _aevents(invocation, _start_event(), None), invocation, False
    )

    proxy._fail(RuntimeError("caller failed"))

    assert context.get_current() is before
    _assert_all_in_invocation(finish_seen, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
