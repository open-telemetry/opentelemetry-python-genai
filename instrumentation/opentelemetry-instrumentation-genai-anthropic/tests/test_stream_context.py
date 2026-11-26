# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streams must not keep the chat span current while they are unconsumed."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib
from anthropic import Anthropic, AsyncAnthropic, NotFoundError
from anthropic.types import (
    Message,
    RawMessageStartEvent,
    Usage,
)

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.anthropic import (
    AnthropicInstrumentor,
)
from opentelemetry.instrumentation.genai.anthropic._raw_response import (
    wrap_raw_response,
)
from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamWrapper,
    MessagesStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.test_util_genai.content import get_content_attribute
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode

_MODEL = "claude-sonnet-4-20250514"
_MESSAGES = [{"role": "user", "content": "Say hello in one word."}]
# The message id the ``test_sync_messages_stream`` cassette replays.
_STREAM_MESSAGE_ID = "msg_011CgwjfMs7kzbGi4KwkwpwR"
_SYNC_STREAM_TEXT = "Hello!"
# ... and the one ``test_async_messages_stream`` replays.
_ASYNC_STREAM_MESSAGE_ID = "msg_01N2EGWxw2zHUcjTjToMivN6"
_ASYNC_STREAM_TEXT = "Hello."


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


def _invocation(
    tracer_provider, logger_provider, meter_provider, completion_hook=None
) -> Any:
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        instrumentation_scope_name=(
            "opentelemetry.instrumentation.genai.anthropic"
        ),
        completion_hook=completion_hook,
    ).inference("anthropic", request_model=_MODEL, _attach_to_context=False)


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


# --- MessageStream helpers ---------------------------------------------------

_HELPERS = ["until_done", "get_final_message", "get_final_text"]


class _HelperStream:
    """Stands in for the SDK's ``MessageStream``, whose helpers drive the SDK
    object directly instead of the wrapper."""

    def __init__(self, invocation: Any) -> None:
        self._invocation = invocation
        self._drained = False
        self.text_stream = self._texts()

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def close(self) -> None:
        pass

    def _texts(self) -> Iterator[str]:
        for text in ("hello", " world"):
            _assert_invocation_current(self._invocation)
            yield text

    def until_done(self) -> None:
        _assert_invocation_current(self._invocation)
        self._drained = True

    # Like the SDK's, these read nothing once ``until_done()`` drained the
    # stream, so the wrapper drains first and calls them outside the span.
    def get_final_message(self) -> Message:
        assert self._drained
        return _start_event().message

    def get_final_text(self) -> str:
        assert self._drained
        return "Hello"


class _AsyncHelperStream:
    def __init__(self, invocation: Any) -> None:
        self._invocation = invocation
        self._drained = False
        self.text_stream = self._texts()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        return
        yield

    async def close(self) -> None:
        pass

    async def _texts(self) -> AsyncIterator[str]:
        for text in ("hello", " world"):
            await asyncio.sleep(0)
            _assert_invocation_current(self._invocation)
            yield text

    async def until_done(self) -> None:
        await asyncio.sleep(0)
        _assert_invocation_current(self._invocation)
        self._drained = True

    async def get_final_message(self) -> Message:
        await asyncio.sleep(0)
        assert self._drained
        return _start_event().message

    async def get_final_text(self) -> str:
        await asyncio.sleep(0)
        assert self._drained
        return "Hello"


def test_sync_text_stream_activates_per_chunk(
    tracer_provider, logger_provider, meter_provider
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(
        _HelperStream(invocation), invocation, False
    )

    texts = stream.text_stream
    assert stream.text_stream is texts
    collected = []
    for text in texts:
        assert context.get_current() is before
        collected.append(text)
    assert collected == ["hello", " world"]
    assert context.get_current() is before


@pytest.mark.parametrize("helper", _HELPERS)
def test_sync_stream_helper_activates(
    tracer_provider, logger_provider, meter_provider, helper
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(
        _HelperStream(invocation), invocation, False
    )

    getattr(stream, helper)()
    assert context.get_current() is before


@pytest.mark.asyncio
async def test_async_text_stream_activates_per_chunk(
    tracer_provider, logger_provider, meter_provider
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = AsyncMessagesStreamWrapper(
        _AsyncHelperStream(invocation), invocation, False
    )

    texts = stream.text_stream
    assert stream.text_stream is texts

    async def drain() -> list[str]:
        collected = []
        async for text in texts:
            assert context.get_current() is before
            collected.append(text)
        return collected

    assert await asyncio.create_task(drain()) == ["hello", " world"]
    assert context.get_current() is before


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", _HELPERS)
async def test_async_stream_helper_activates(
    tracer_provider, logger_provider, meter_provider, helper
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = AsyncMessagesStreamWrapper(
        _AsyncHelperStream(invocation), invocation, False
    )

    await asyncio.create_task(getattr(stream, helper)())
    assert context.get_current() is before


# --- the scoped text stream closed early by the caller -----------------------


class _ClosableTextStream:
    """A ``MessageStream`` stand-in whose ``text_stream`` is a real generator.

    The SDK hands the caller a generator, so an adapter closed after one chunk
    has to close that generator; it records the span it is closed under.
    """

    current_message_snapshot = _start_event().message

    def __init__(
        self, invocation: Any, closed: list[trace.SpanContext]
    ) -> None:
        self._invocation = invocation
        self._closed = closed
        self.text_stream = self._texts()

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def close(self) -> None:
        pass

    def _texts(self) -> Iterator[str]:
        try:
            _assert_invocation_current(self._invocation)
            yield "hello"
            _assert_invocation_current(self._invocation)
            yield " world"
        finally:
            self._closed.append(trace.get_current_span().get_span_context())


class _AsyncClosableTextStream:
    """Async counterpart of ``_ClosableTextStream``."""

    current_message_snapshot = _start_event().message

    def __init__(
        self, invocation: Any, closed: list[trace.SpanContext]
    ) -> None:
        self._invocation = invocation
        self._closed = closed
        self.text_stream = self._texts()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        return
        yield

    async def close(self) -> None:
        pass

    async def _texts(self) -> AsyncIterator[str]:
        try:
            await asyncio.sleep(0)
            _assert_invocation_current(self._invocation)
            yield "hello"
            await asyncio.sleep(0)
            _assert_invocation_current(self._invocation)
            yield " world"
        finally:
            self._closed.append(trace.get_current_span().get_span_context())


def test_sync_text_stream_early_close_closes_the_sdk_generator(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    closed: list[trace.SpanContext] = []
    source = _ClosableTextStream(invocation, closed)
    stream = MessagesStreamWrapper(source, invocation, False)

    texts = stream.text_stream
    assert next(texts) == "hello"
    assert context.get_current() is before

    texts.close()

    assert inspect.getgeneratorstate(source.text_stream) == inspect.GEN_CLOSED
    _assert_all_in_invocation(closed, invocation)
    assert context.get_current() is before
    stream.close()
    (span,) = span_exporter.get_finished_spans()
    assert "error.type" not in (span.attributes or {})


@pytest.mark.asyncio
async def test_async_text_stream_early_close_closes_the_sdk_generator(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    closed: list[trace.SpanContext] = []
    source = _AsyncClosableTextStream(invocation, closed)
    stream = AsyncMessagesStreamWrapper(source, invocation, False)

    texts = stream.text_stream
    assert await anext(texts) == "hello"
    assert context.get_current() is before

    await texts.aclose()

    assert source.text_stream.ag_frame is None
    _assert_all_in_invocation(closed, invocation)
    assert context.get_current() is before
    await stream.close()
    (span,) = span_exporter.get_finished_spans()
    assert "error.type" not in (span.attributes or {})


class _FailingCloseTextStream(_ClosableTextStream):
    def _texts(self) -> Iterator[str]:
        try:
            yield "hello"
            yield " world"
        finally:
            raise RuntimeError("sdk close failed")


class _AsyncFailingCloseTextStream(_AsyncClosableTextStream):
    async def _texts(self) -> AsyncIterator[str]:
        try:
            yield "hello"
            yield " world"
        finally:
            raise RuntimeError("sdk close failed")


def test_sync_text_stream_close_error_propagates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    # Closing the SDK's own generator raises the same error uninstrumented.
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(
        _FailingCloseTextStream(invocation, []), invocation, False
    )

    texts = stream.text_stream
    assert next(texts) == "hello"
    with pytest.raises(RuntimeError, match="sdk close failed"):
        texts.close()

    assert context.get_current() is before
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_1"
    stream.close()
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_text_stream_close_error_propagates(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = AsyncMessagesStreamWrapper(
        _AsyncFailingCloseTextStream(invocation, []), invocation, False
    )

    texts = stream.text_stream
    assert await anext(texts) == "hello"
    with pytest.raises(RuntimeError, match="sdk close failed"):
        await texts.aclose()

    assert context.get_current() is before
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "RuntimeError"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_1"
    await stream.close()
    assert len(span_exporter.get_finished_spans()) == 1


def test_stream_helpers_not_advertised_on_plain_streams(
    tracer_provider, logger_provider, meter_provider
) -> None:
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(iter(()), invocation, False)
    async_stream = AsyncMessagesStreamWrapper(
        _aevents(invocation, _start_event(), None), invocation, False
    )

    for name in ["text_stream", *_HELPERS]:
        assert not hasattr(stream, name)
        assert not hasattr(async_stream, name)


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_stream")
def test_sync_messages_stream_helpers_keep_caller_span_current(
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
            for _ in stream.text_stream:
                assert _is_current(caller)
            message = stream.get_final_message()
            assert _is_current(caller)
        assert _is_current(caller)
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    assert chat.parent is not None
    assert chat.parent.span_id == caller.get_span_context().span_id
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == message.id
    _assert_no_detach_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_stream")
async def test_async_messages_stream_helpers_keep_caller_span_current(
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
            async for _ in stream.text_stream:
                assert _is_current(caller)
            message = await stream.get_final_message()
            assert _is_current(caller)
        assert _is_current(caller)
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    assert chat.parent is not None
    assert chat.parent.span_id == caller.get_span_context().span_id
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == message.id
    _assert_no_detach_errors(caplog)


# --- MessageStream helpers, one fresh SDK stream per helper -------------------


def _assert_chat_under_caller(
    span_exporter, caller: Any, caplog, request_model: str = _MODEL
) -> Any:
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    assert chat.parent is not None
    assert chat.parent.span_id == caller.get_span_context().span_id
    assert (
        chat.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )
    _assert_no_detach_errors(caplog)
    return chat


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_stream")
@pytest.mark.parametrize("helper", _HELPERS)
def test_sync_messages_stream_helper_read_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
    helper,
):
    """Each draining helper reads a fresh SDK stream under the invocation."""
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with anthropic_client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            result = getattr(stream, helper)()
            assert _is_current(caller)
            if helper == "get_final_text":
                assert result == _SYNC_STREAM_TEXT
        assert _is_current(caller)
    chat = _assert_chat_under_caller(span_exporter, caller, caplog)
    assert chat.status.status_code is not trace.StatusCode.ERROR
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == (
        _STREAM_MESSAGE_ID
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_stream")
@pytest.mark.parametrize("helper", _HELPERS)
async def test_async_messages_stream_helper_read_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
    caplog,
    helper,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        async with async_anthropic_client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            result = await asyncio.create_task(getattr(stream, helper)())
            assert _is_current(caller)
            if helper == "get_final_text":
                assert result == _ASYNC_STREAM_TEXT
        assert _is_current(caller)
    chat = _assert_chat_under_caller(span_exporter, caller, caplog)
    assert chat.status.status_code is not trace.StatusCode.ERROR
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == (
        _ASYNC_STREAM_MESSAGE_ID
    )


# --- helpers reading a public stream, observed per chunk ---------------------


def _sse(event: str, data: dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


_SSE_TEXT = "Hi"
_SSE_CHUNKS = [
    _sse(
        "message_start",
        {
            "type": "message_start",
            "message": _start_event().message.model_dump(),
        },
    ),
    _sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
    ),
    _sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": _SSE_TEXT},
        },
    ),
    _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
    _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
    ),
    _sse("message_stop", {"type": "message_stop"}),
]
# Fail after ``message_start``, so the SDK snapshot already holds the message.
_FAIL_AFTER = 1
# Fail after the text delta, so the snapshot also holds the partial text.
_FAIL_AFTER_TEXT = 3


class _ObservedSSE(_http_lib.SyncByteStream):
    """Serves the SSE body, recording the current span at every chunk read."""

    def __init__(
        self, seen: list[trace.SpanContext], fail_at: int | None
    ) -> None:
        self._seen = seen
        self._fail_at = fail_at

    def __iter__(self) -> Iterator[bytes]:
        for index, chunk in enumerate(_SSE_CHUNKS):
            self._seen.append(trace.get_current_span().get_span_context())
            if index == self._fail_at:
                raise ConnectionError("connection dropped")
            yield chunk


class _AsyncObservedSSE(_http_lib.AsyncByteStream):
    def __init__(
        self, seen: list[trace.SpanContext], fail_at: int | None
    ) -> None:
        self._seen = seen
        self._fail_at = fail_at

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(_SSE_CHUNKS):
            await asyncio.sleep(0)
            self._seen.append(trace.get_current_span().get_span_context())
            if index == self._fail_at:
                raise ConnectionError("connection dropped")
            yield chunk


def _sse_response(stream: Any) -> Any:
    return _http_lib.Response(
        200, headers={"content-type": "text/event-stream"}, stream=stream
    )


def _observed_client(
    seen: list[trace.SpanContext], fail_at: int | None = None
) -> Anthropic:
    return Anthropic(
        api_key="test_anthropic_api_key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=_http_lib.Client(
            transport=_http_lib.MockTransport(
                lambda request: _sse_response(_ObservedSSE(seen, fail_at))
            )
        ),
    )


def _async_observed_client(
    seen: list[trace.SpanContext], fail_at: int | None = None
) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key="test_anthropic_api_key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=_http_lib.AsyncClient(
            transport=_http_lib.MockTransport(
                lambda request: _sse_response(_AsyncObservedSSE(seen, fail_at))
            )
        ),
    )


def _drain_text(stream: Any) -> None:
    for _ in stream.text_stream:
        pass


async def _adrain_text(stream: Any) -> None:
    async for _ in stream.text_stream:
        pass


_SYNC_CONSUMERS = {
    **{
        name: (lambda stream, name=name: getattr(stream, name)())
        for name in _HELPERS
    },
    "text_stream": _drain_text,
}
_ASYNC_CONSUMERS = {
    **{
        name: (lambda stream, name=name: getattr(stream, name)())
        for name in _HELPERS
    },
    "text_stream": _adrain_text,
}


def _assert_reads_under_chat(
    span_exporter, caller: Any, seen: list[trace.SpanContext]
) -> Any:
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    assert chat.parent is not None
    assert chat.parent.span_id == caller.get_span_context().span_id
    assert seen
    assert all(
        span_context == chat.get_span_context() for span_context in seen
    )
    return chat


def _assert_dropped_connection(chat: Any) -> None:
    assert chat.status.status_code is trace.StatusCode.ERROR
    assert chat.attributes is not None
    assert chat.attributes["error.type"] == "ConnectionError"
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_1"


@pytest.mark.parametrize("fail", [False, True], ids=["drained", "dropped"])
@pytest.mark.parametrize("consumer", list(_SYNC_CONSUMERS))
def test_sync_messages_stream_helper_reads_run_under_the_chat_span(
    tracer_provider, span_exporter, instrument_no_content, consumer, fail
) -> None:
    """Every chunk a helper reads from a public stream is read under ``chat``."""
    seen: list[trace.SpanContext] = []
    client = _observed_client(seen, _FAIL_AFTER if fail else None)
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            if fail:
                with pytest.raises(ConnectionError, match="dropped"):
                    _SYNC_CONSUMERS[consumer](stream)
            else:
                _SYNC_CONSUMERS[consumer](stream)
            assert _is_current(caller)
    client.close()

    chat = _assert_reads_under_chat(span_exporter, caller, seen)
    if fail:
        _assert_dropped_connection(chat)
    else:
        assert chat.status.status_code is not trace.StatusCode.ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True], ids=["drained", "dropped"])
@pytest.mark.parametrize("consumer", list(_ASYNC_CONSUMERS))
async def test_async_messages_stream_helper_reads_run_under_the_chat_span(
    tracer_provider, span_exporter, instrument_no_content, consumer, fail
) -> None:
    seen: list[trace.SpanContext] = []
    client = _async_observed_client(seen, _FAIL_AFTER if fail else None)
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        async with client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            if fail:
                with pytest.raises(ConnectionError, match="dropped"):
                    await asyncio.create_task(
                        _ASYNC_CONSUMERS[consumer](stream)
                    )
            else:
                await asyncio.create_task(_ASYNC_CONSUMERS[consumer](stream))
            assert _is_current(caller)
    await client.close()

    chat = _assert_reads_under_chat(span_exporter, caller, seen)
    if fail:
        _assert_dropped_connection(chat)
    else:
        assert chat.status.status_code is not trace.StatusCode.ERROR


@pytest.fixture
def instrument_content_capture(
    tracer_provider, logger_provider, meter_provider, content_capture
) -> Any:
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture=content_capture.name,
    ):
        yield content_capture


def _assert_partial_output(chat: Any, mode: ContentCapturingMode) -> None:
    assert chat.attributes is not None
    if mode is ContentCapturingMode.SPAN_ONLY:
        (message,) = get_content_attribute(
            chat.attributes, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
        )
        assert message["role"] == "assistant"
        assert message["parts"] == [{"type": "text", "content": _SSE_TEXT}]
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in chat.attributes


@pytest.mark.parametrize("consumer", list(_SYNC_CONSUMERS))
def test_sync_messages_stream_helper_failure_after_text_keeps_partial_output(
    span_exporter, instrument_content_capture, consumer
) -> None:
    client = _observed_client([], _FAIL_AFTER_TEXT)
    with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        with pytest.raises(ConnectionError, match="dropped"):
            _SYNC_CONSUMERS[consumer](stream)
    client.close()

    (chat,) = span_exporter.get_finished_spans()
    _assert_dropped_connection(chat)
    _assert_partial_output(chat, instrument_content_capture)


@pytest.mark.asyncio
@pytest.mark.parametrize("consumer", list(_ASYNC_CONSUMERS))
async def test_async_messages_stream_helper_failure_after_text_keeps_partial_output(
    span_exporter, instrument_content_capture, consumer
) -> None:
    client = _async_observed_client([], _FAIL_AFTER_TEXT)
    async with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        with pytest.raises(ConnectionError, match="dropped"):
            await _ASYNC_CONSUMERS[consumer](stream)
    await client.close()

    (chat,) = span_exporter.get_finished_spans()
    _assert_dropped_connection(chat)
    _assert_partial_output(chat, instrument_content_capture)


def test_sync_messages_stream_text_stream_close_before_read_closes_the_sdk_generator(
    tracer_provider, instrument_no_content
) -> None:
    client = _observed_client([])
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            source = stream.__wrapped__.text_stream
            stream.text_stream.close()

            assert inspect.getgeneratorstate(source) == inspect.GEN_CLOSED
            assert _is_current(caller)
    client.close()


@pytest.mark.asyncio
async def test_async_messages_stream_text_stream_close_before_read_closes_the_sdk_generator(
    tracer_provider, instrument_no_content
) -> None:
    client = _async_observed_client([])
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        async with client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            source = stream.__wrapped__.text_stream
            await stream.text_stream.aclose()

            assert source.ag_frame is None
            assert _is_current(caller)
    await client.close()


def _chat_span(span_exporter) -> Any:
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    return chat


@pytest.mark.parametrize("helper", _HELPERS)
def test_sync_messages_stream_helper_rejects_arguments_before_reading(
    span_exporter, instrument_no_content, helper
) -> None:
    """The SDK helpers take no arguments, so a bad call reads nothing."""
    seen: list[trace.SpanContext] = []
    client = _observed_client(seen)
    with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        read_before = len(seen)
        with pytest.raises(TypeError, match="unexpected"):
            getattr(stream, helper)(unexpected=True)
        assert len(seen) == read_before
        assert span_exporter.get_finished_spans() == ()

        getattr(stream, helper)()
        assert len(seen) > read_before
    client.close()

    assert _chat_span(span_exporter).status.status_code is not (
        trace.StatusCode.ERROR
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", _HELPERS)
async def test_async_messages_stream_helper_rejects_arguments_when_called(
    span_exporter, instrument_no_content, helper
) -> None:
    """A bad call raises when made, as the SDK's own coroutine function does."""
    seen: list[trace.SpanContext] = []
    client = _async_observed_client(seen)
    async with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        read_before = len(seen)
        with pytest.raises(TypeError, match="unexpected"):
            getattr(stream, helper)(unexpected=True)
        assert len(seen) == read_before
        assert span_exporter.get_finished_spans() == ()

        await getattr(stream, helper)()
        assert len(seen) > read_before
    await client.close()

    assert _chat_span(span_exporter).status.status_code is not (
        trace.StatusCode.ERROR
    )


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_stream_api_error")
def test_sync_messages_stream_open_error_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
):
    """An SDK error opening the stream fails the span under the caller's span."""
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with pytest.raises(NotFoundError):
            with anthropic_client.messages.stream(
                model="invalid-model-name",
                max_tokens=100,
                messages=_MESSAGES,
            ) as stream:
                list(stream)
        assert _is_current(caller)
    chat = _assert_chat_under_caller(
        span_exporter, caller, caplog, request_model="invalid-model-name"
    )
    assert chat.status.status_code is trace.StatusCode.ERROR
    assert chat.attributes is not None
    assert "NotFoundError" in chat.attributes["error.type"]


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_stream_api_error")
async def test_async_messages_stream_open_error_keeps_caller_span_current(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_no_content,
    caplog,
):
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with pytest.raises(NotFoundError):
            async with async_anthropic_client.messages.stream(
                model="invalid-model-name",
                max_tokens=100,
                messages=_MESSAGES,
            ) as stream:
                async for _ in stream:
                    pass
        assert _is_current(caller)
    chat = _assert_chat_under_caller(
        span_exporter, caller, caplog, request_model="invalid-model-name"
    )
    assert chat.status.status_code is trace.StatusCode.ERROR
    assert chat.attributes is not None
    assert "NotFoundError" in chat.attributes["error.type"]


# --- final-message helpers on a successful response --------------------------

_TOOL_USE_SSE = b"".join(
    [
        _SSE_CHUNKS[0],
        _sse(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {},
                },
            },
        ),
        _sse(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    ]
)


def _tool_use_response(request: Any) -> Any:
    return _http_lib.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_TOOL_USE_SSE,
    )


def _assert_tool_use_span_succeeded(span_exporter) -> None:
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is not trace.StatusCode.ERROR
    assert span.attributes is not None
    assert "error.type" not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )


def test_sync_get_final_text_without_text_block_keeps_the_span_ok(
    span_exporter, instrument_no_content
) -> None:
    """The SDK's usage error on a drained, successful stream is the caller's."""
    client = Anthropic(
        api_key="test_anthropic_api_key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=_http_lib.Client(
            transport=_http_lib.MockTransport(_tool_use_response)
        ),
    )
    with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        with pytest.raises(RuntimeError, match="text`? (content )?block"):
            stream.get_final_text()
    client.close()

    _assert_tool_use_span_succeeded(span_exporter)


@pytest.mark.asyncio
async def test_async_get_final_text_without_text_block_keeps_the_span_ok(
    span_exporter, instrument_no_content
) -> None:
    client = AsyncAnthropic(
        api_key="test_anthropic_api_key",
        base_url="http://anthropic.test",
        max_retries=0,
        http_client=_http_lib.AsyncClient(
            transport=_http_lib.MockTransport(_tool_use_response)
        ),
    )
    async with client.messages.stream(
        model=_MODEL, max_tokens=100, messages=_MESSAGES
    ) as stream:
        with pytest.raises(RuntimeError, match="text`? (content )?block"):
            await stream.get_final_text()
    await client.close()

    _assert_tool_use_span_succeeded(span_exporter)


# --- the SDK text stream closed early through the public API ------------------


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_stream")
def test_sync_messages_stream_text_stream_closed_early_by_caller(
    tracer_provider,
    span_exporter,
    anthropic_client,
    instrument_no_content,
    caplog,
):
    """Closing ``text_stream`` after one chunk closes the SDK generator too."""
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("caller") as caller:
        with anthropic_client.messages.stream(
            model=_MODEL, max_tokens=100, messages=_MESSAGES
        ) as stream:
            sdk_stream = stream.__wrapped__
            texts = stream.text_stream
            assert next(texts) == _SYNC_STREAM_TEXT
            texts.close()
            # The generator the SDK handed out is closed rather than left
            # suspended at the chunk the caller stopped after.
            assert (
                inspect.getgeneratorstate(sdk_stream.text_stream)
                == inspect.GEN_CLOSED
            )
            assert _is_current(caller)
        assert _is_current(caller)
    chat = _assert_chat_under_caller(span_exporter, caller, caplog)
    assert chat.status.status_code is not trace.StatusCode.ERROR
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == (
        _STREAM_MESSAGE_ID
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_stream")
async def test_async_messages_stream_text_stream_closed_early_by_caller(
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
            sdk_stream = stream.__wrapped__
            texts = stream.text_stream
            assert await anext(texts) == _ASYNC_STREAM_TEXT
            await texts.aclose()
            assert sdk_stream.text_stream.ag_frame is None
            assert _is_current(caller)
        assert _is_current(caller)
    chat = _assert_chat_under_caller(span_exporter, caller, caplog)
    assert chat.status.status_code is not trace.StatusCode.ERROR
    assert chat.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == (
        _ASYNC_STREAM_MESSAGE_ID
    )


# --- closing through stream.response -----------------------------------------


class _ObservedResponse:
    def __init__(self, seen: list[trace.SpanContext]) -> None:
        self._seen = seen

    def close(self) -> None:
        self._seen.append(trace.get_current_span().get_span_context())

    async def aclose(self) -> None:
        await asyncio.sleep(0)
        self._seen.append(trace.get_current_span().get_span_context())


class _StreamWithResponse:
    def __init__(self, seen: list[trace.SpanContext]) -> None:
        self.response = _ObservedResponse(seen)

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __aiter__(self) -> AsyncIterator[Any]:
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        return
        yield

    def close(self) -> None:
        pass


def test_sync_response_close_finalizes_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    hook = _ContextRecordingHook()
    invocation = _invocation(
        tracer_provider, logger_provider, meter_provider, hook
    )
    seen: list[trace.SpanContext] = []
    stream = MessagesStreamWrapper(
        _StreamWithResponse(seen), invocation, False
    )

    stream.response.close()

    assert context.get_current() is before
    assert len(seen) == 1
    _assert_all_in_invocation(seen + hook.contexts, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_async_response_aclose_finalizes_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    hook = _ContextRecordingHook()
    invocation = _invocation(
        tracer_provider, logger_provider, meter_provider, hook
    )
    seen: list[trace.SpanContext] = []
    stream = AsyncMessagesStreamWrapper(
        _StreamWithResponse(seen), invocation, False
    )

    await asyncio.create_task(stream.response.aclose())

    assert context.get_current() is before
    assert len(seen) == 1
    _assert_all_in_invocation(seen + hook.contexts, invocation)
    assert len(span_exporter.get_finished_spans()) == 1


def test_sync_raw_body_iteration_after_span_end_keeps_caller_context(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    # What a stream wrapper leaves behind: the span has ended and the wrapper
    # owns it. Body iteration must not put it back in the caller's context.
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedStream(seen))
    proxy._self_span_open = False

    for _ in proxy.http_response.iter_raw():
        assert context.get_current() is before

    assert context.get_current() is before
    assert seen
    assert not any(span_context.is_valid for span_context in seen)
    assert span_exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_async_raw_body_iteration_after_span_end_keeps_caller_context(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    seen: list[trace.SpanContext] = []
    proxy = _body_proxy(invocation, _ObservedAsyncStream(seen))
    proxy._self_span_open = False

    async def drain() -> None:
        async for _ in proxy.http_response.aiter_raw():
            assert context.get_current() is before

    await asyncio.create_task(drain())

    assert context.get_current() is before
    assert seen
    assert not any(span_context.is_valid for span_context in seen)
    assert span_exporter.get_finished_spans() == ()


# --- raw response, parse() returns a Message ---------------------------------


class _ParsingRaw:
    def __init__(self, response: Any) -> None:
        self.http_response = response

    def parse(self) -> Message:
        return _start_event().message


class _AsyncParsingRaw:
    def __init__(self, response: Any) -> None:
        self.http_response = response

    async def parse(self) -> Message:
        await asyncio.sleep(0)
        return _start_event().message


def test_sync_raw_response_parse_message_stops_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    hook = _ContextRecordingHook()
    invocation = _invocation(
        tracer_provider, logger_provider, meter_provider, hook
    )
    response = _http_lib.Response(
        200, stream=_ObservedStream([]), request=_REQUEST
    )
    proxy = wrap_raw_response(_ParsingRaw(response), invocation, False)

    message = proxy.parse()

    assert isinstance(message, Message)
    assert context.get_current() is before
    _assert_all_in_invocation(hook.contexts, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == message.id


@pytest.mark.asyncio
async def test_async_raw_response_parse_message_stops_inside_invocation(
    tracer_provider, logger_provider, meter_provider, span_exporter
) -> None:
    before = context.get_current()
    hook = _ContextRecordingHook()
    invocation = _invocation(
        tracer_provider, logger_provider, meter_provider, hook
    )
    response = _http_lib.Response(
        200, stream=_ObservedAsyncStream([]), request=_REQUEST
    )
    proxy = wrap_raw_response(_AsyncParsingRaw(response), invocation, False)

    message = await asyncio.create_task(proxy.parse())

    assert isinstance(message, Message)
    assert context.get_current() is before
    _assert_all_in_invocation(hook.contexts, invocation)
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == message.id


# --- SDK errors in the scoped work -------------------------------------------


class _FailingHelperStream(_HelperStream):
    """A ``MessageStream`` whose helpers fail after part of the response
    arrived, so ``current_message_snapshot`` already holds it."""

    current_message_snapshot = _start_event().message

    def until_done(self) -> None:
        raise ConnectionError("helper failed")

    def _texts(self) -> Iterator[str]:
        yield "hello"
        raise ConnectionError("helper failed")


class _FailingAsyncHelperStream(_AsyncHelperStream):
    current_message_snapshot = _start_event().message

    async def until_done(self) -> None:
        raise ConnectionError("helper failed")

    async def _texts(self) -> AsyncIterator[str]:
        yield "hello"
        raise ConnectionError("helper failed")


def _assert_failed_with_snapshot(span_exporter) -> None:
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code is trace.StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ConnectionError"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_MODEL] == _MODEL
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_1"


def _drain_sync_text(stream: Any) -> None:
    for _ in stream.text_stream:
        pass


async def _drain_async_text(stream: Any) -> None:
    async for _ in stream.text_stream:
        pass


@pytest.mark.parametrize(
    "consume",
    [lambda stream: stream.get_final_message(), _drain_sync_text],
    ids=["get_final_message", "text_stream"],
)
def test_sync_stream_helper_error_fails_the_span(
    tracer_provider, logger_provider, meter_provider, span_exporter, consume
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = MessagesStreamWrapper(
        _FailingHelperStream(invocation), invocation, False
    )

    with pytest.raises(ConnectionError, match="helper failed"):
        consume(stream)

    assert context.get_current() is before
    _assert_failed_with_snapshot(span_exporter)
    # Already finalized: a later close must not end the span again.
    stream.close()
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "consume",
    [lambda stream: stream.get_final_message(), _drain_async_text],
    ids=["get_final_message", "text_stream"],
)
async def test_async_stream_helper_error_fails_the_span(
    tracer_provider, logger_provider, meter_provider, span_exporter, consume
) -> None:
    before = context.get_current()
    invocation = _invocation(tracer_provider, logger_provider, meter_provider)
    stream = AsyncMessagesStreamWrapper(
        _FailingAsyncHelperStream(invocation), invocation, False
    )

    with pytest.raises(ConnectionError, match="helper failed"):
        await asyncio.create_task(consume(stream))

    assert context.get_current() is before
    _assert_failed_with_snapshot(span_exporter)
    await stream.close()
    assert len(span_exporter.get_finished_spans()) == 1


# --- raw response, public error paths ----------------------------------------


class _ContextRecordingHook:
    """Completion hook that records the span finalization runs under."""

    def __init__(self) -> None:
        self.contexts: list[trace.SpanContext] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.contexts.append(trace.get_current_span().get_span_context())


@pytest.fixture
def instrument_with_hook(
    tracer_provider, logger_provider, meter_provider
) -> Any:
    """Instrument with a completion hook that records its span."""
    hook = _ContextRecordingHook()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
        completion_hook=hook,
    ):
        yield hook


def _assert_hook_ran_under(hook: _ContextRecordingHook, span: Any) -> None:
    assert hook.contexts
    assert all(
        span_context == span.get_span_context()
        for span_context in hook.contexts
    )


def _assert_raw_response_caller_error(
    span_exporter, caller: Any, hook: _ContextRecordingHook
) -> None:
    (chat,) = [
        s for s in span_exporter.get_finished_spans() if s.name != "caller"
    ]
    assert chat.status.status_code is trace.StatusCode.ERROR
    assert chat.attributes is not None
    assert chat.attributes["error.type"] == "RuntimeError"
    assert chat.parent is not None
    assert chat.parent.span_id == caller.get_span_context().span_id
    _assert_hook_ran_under(hook, chat)


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_create_streaming_with_raw_response")
def test_sync_streaming_response_error_before_parse_fails_the_span(
    tracer_provider, span_exporter, anthropic_client, instrument_with_hook
) -> None:
    """A caller error inside the response block fails the span."""
    before = context.get_current()
    tracer = tracer_provider.get_tracer(__name__)
    with pytest.raises(RuntimeError, match="caller failed"):
        with tracer.start_as_current_span("caller") as caller:
            with anthropic_client.messages.with_streaming_response.create(
                model=_MODEL,
                max_tokens=100,
                messages=_MESSAGES,
                stream=True,
            ):
                assert _is_current(caller)
                raise RuntimeError("caller failed")
    assert context.get_current() is before
    _assert_raw_response_caller_error(
        span_exporter, caller, instrument_with_hook
    )


@pytest.mark.vcr()
@pytest.mark.cassette("test_sync_messages_create_streaming_with_raw_response")
def test_sync_streaming_response_error_after_parse_fails_the_span(
    tracer_provider, span_exporter, anthropic_client, instrument_with_hook
) -> None:
    """A caller error after ``parse()`` fails the stream's span."""
    before = context.get_current()
    tracer = tracer_provider.get_tracer(__name__)
    with pytest.raises(RuntimeError, match="caller failed"):
        with tracer.start_as_current_span("caller") as caller:
            with anthropic_client.messages.with_streaming_response.create(
                model=_MODEL,
                max_tokens=100,
                messages=_MESSAGES,
                stream=True,
            ) as raw_response:
                assert isinstance(raw_response.parse(), MessagesStreamWrapper)
                assert _is_current(caller)
                raise RuntimeError("caller failed")
    assert context.get_current() is before
    _assert_raw_response_caller_error(
        span_exporter, caller, instrument_with_hook
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_create_streaming_with_raw_response")
async def test_async_streaming_response_error_before_parse_fails_the_span(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_with_hook,
) -> None:
    before = context.get_current()
    tracer = tracer_provider.get_tracer(__name__)
    with pytest.raises(RuntimeError, match="caller failed"):
        with tracer.start_as_current_span("caller") as caller:
            async with (
                async_anthropic_client.messages.with_streaming_response.create(
                    model=_MODEL,
                    max_tokens=100,
                    messages=_MESSAGES,
                    stream=True,
                )
            ):
                assert _is_current(caller)
                raise RuntimeError("caller failed")
    assert context.get_current() is before
    _assert_raw_response_caller_error(
        span_exporter, caller, instrument_with_hook
    )


@pytest.mark.asyncio
@pytest.mark.vcr()
@pytest.mark.cassette("test_async_messages_create_streaming_with_raw_response")
async def test_async_streaming_response_error_after_parse_fails_the_span(
    tracer_provider,
    span_exporter,
    async_anthropic_client,
    instrument_with_hook,
) -> None:
    before = context.get_current()
    tracer = tracer_provider.get_tracer(__name__)
    with pytest.raises(RuntimeError, match="caller failed"):
        with tracer.start_as_current_span("caller") as caller:
            async with (
                async_anthropic_client.messages.with_streaming_response.create(
                    model=_MODEL,
                    max_tokens=100,
                    messages=_MESSAGES,
                    stream=True,
                )
            ) as raw_response:
                stream = raw_response.parse()
                if inspect.isawaitable(stream):
                    stream = await stream
                assert isinstance(stream, AsyncMessagesStreamWrapper)
                assert _is_current(caller)
                raise RuntimeError("caller failed")
    assert context.get_current() is before
    _assert_raw_response_caller_error(
        span_exporter, caller, instrument_with_hook
    )
