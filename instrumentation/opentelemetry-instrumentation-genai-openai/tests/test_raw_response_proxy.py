# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the streaming ``with_raw_response`` proxy.

These exercise ``RawResponseStreamProxy.parse()`` — it wraps only SDK streams
it can drive, awaits the coroutine the async client's ``parse()`` returns, and
hands anything else back untouched — and the close fallback that finalizes the
span when the caller never parses, so the span never leaks.
"""

import asyncio
import inspect
import logging

import pytest
from openai import AsyncStream, Stream

from opentelemetry.instrumentation.genai.openai._raw_response import (
    RawResponseStreamProxy,
    wrap_stream_result,
)
from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    ChatStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class _HttpResponse:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1

    async def aclose(self):
        self.close_calls += 1


class _SyncWrapper:
    """Stands in for a sync stream wrapper, which finalizes on close()."""

    def __init__(self, stream=None):
        self.stream = stream
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakeStream(Stream):
    # Bypass Stream.__init__ (needs an httpx response + client); the proxy only
    # checks isinstance to decide whether it can wrap the parse result.
    def __init__(self):
        pass


class _FakeAsyncStream(AsyncStream):
    # See _FakeStream.
    def __init__(self):
        pass


class _RawResponse:
    headers = {"openai-version": "2020-10-01"}
    request_id = "req_123"

    def __init__(self, parse_result):
        self.http_response = _HttpResponse()
        self._parse_result = parse_result

    def parse(self, *, to=None):
        return self._parse_result


class _AsyncRawResponse(_RawResponse):
    """Shaped like ``AsyncAPIResponse``, whose ``parse()`` is a coroutine."""

    async def parse(self, *, to=None):
        return self._parse_result


def _noop() -> None:
    pass


@pytest.fixture(autouse=True)
def fixture_vcr():
    """No VCR needed for these unit tests."""
    yield


def test_parse_wraps_stream_and_memoizes():
    stream = _FakeStream()
    raw = _RawResponse(stream)
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=_noop,
    )

    # Metadata resolves natively off the raw response.
    assert proxy.headers == {"openai-version": "2020-10-01"}
    assert proxy.request_id == "req_123"

    parsed = proxy.parse()
    assert parsed.stream is stream
    # Memoized so repeated calls share one wrapper / span.
    assert proxy.parse() is parsed


def test_parse_non_stream_returned_untouched(caplog):
    # parse() may return something we can't drive — e.g. a custom non-stream
    # target passed as ``to=``. The proxy hands it back untouched, logs, and
    # does NOT finalize; the close fallback stays armed to finalize on
    # drain/close.
    raw = _RawResponse("not-a-stream")
    wrapped_calls = []
    finalize_calls = []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=lambda s: wrapped_calls.append(s) or _SyncWrapper(s),
        finalize=lambda: finalize_calls.append(True),
    )

    with caplog.at_level(
        logging.DEBUG,
        logger="opentelemetry.instrumentation.genai.openai._raw_response",
    ):
        parsed = proxy.parse()
    assert parsed == "not-a-stream"  # handed back untouched
    assert wrapped_calls == []  # wrap_stream never invoked
    assert finalize_calls == []  # parse must not finalize; the close hook does
    assert "skipping stream instrumentation" in caplog.text

    # Fallback still armed: closing the body finalizes the span.
    raw.http_response.close()
    assert finalize_calls == [True]


@pytest.mark.asyncio()
async def test_parse_awaits_coroutine_and_wraps():
    # The async client's with_streaming_response hands back an AsyncAPIResponse
    # whose parse() is a coroutine. parse() must stay awaitable for the caller
    # and resolve to the instrumented wrapper, not the bare SDK stream.
    stream = _FakeAsyncStream()
    raw = _AsyncRawResponse(stream)
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=_noop,
    )

    pending = proxy.parse()
    assert inspect.isawaitable(pending)  # caller still writes `await parse()`
    assert (await pending).stream is stream


@pytest.mark.asyncio()
async def test_parse_stays_awaitable_once_memoized():
    # Memoizing must not change the shape parse() returns: the caller writes
    # `await parse()` every time, so a memoized hit has to stay awaitable
    # instead of handing back the bare wrapper.
    stream = _FakeAsyncStream()
    raw = _AsyncRawResponse(stream)
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=_noop,
    )

    first = await proxy.parse()

    second = proxy.parse()
    assert inspect.isawaitable(second)
    assert await second is first  # one wrapper / span, shared


@pytest.mark.asyncio()
async def test_parse_awaited_non_stream_returned_untouched(caplog):
    # A coroutine parse() resolving to something we can't drive is handed back
    # untouched, exactly like the synchronous case.
    raw = _AsyncRawResponse("not-a-stream")
    finalize_calls = []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=lambda: finalize_calls.append(True),
    )

    with caplog.at_level(
        logging.DEBUG,
        logger="opentelemetry.instrumentation.genai.openai._raw_response",
    ):
        parsed = await proxy.parse()

    assert parsed == "not-a-stream"
    assert finalize_calls == []  # parse must not finalize; the close hook does
    assert "skipping stream instrumentation" in caplog.text


@pytest.mark.asyncio()
async def test_close_before_awaiting_parse_finalizes_once():
    # parse() returning an un-awaited coroutine leaves the proxy without a
    # wrapper, so the close fallback is still what finalizes the span.
    raw = _AsyncRawResponse(_FakeAsyncStream())
    finalize_calls = []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=lambda: finalize_calls.append(True),
    )

    pending = proxy.parse()
    raw.http_response.close()
    assert finalize_calls == [True]  # span closed, did not leak

    await pending  # tidy up the coroutine so it is not left un-awaited


@pytest.mark.asyncio()
async def test_close_while_parse_pending_does_not_wrap():
    # Same race as above, but the coroutine resolves to a stream we could wrap.
    # The close fallback has already ended the span by then, so a wrapper would
    # accumulate attributes that silently go nowhere -- worse than no wrapper.
    resume = asyncio.Event()
    stream = _FakeAsyncStream()

    class _SlowAsyncRawResponse(_AsyncRawResponse):
        async def parse(self, *, to=None):
            await resume.wait()
            return self._parse_result

    raw = _SlowAsyncRawResponse(stream)
    finalize_calls, wrap_calls = [], []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=lambda s: wrap_calls.append(s) or _SyncWrapper(s),
        finalize=lambda: finalize_calls.append(True),
    )

    pending = proxy.parse()
    raw.http_response.close()
    assert finalize_calls == [True]  # close fallback already finalized

    resume.set()
    result = await pending

    assert wrap_calls == []  # no wrapper built against a finished span
    assert result is stream  # caller still gets a usable stream


def test_close_without_parse_finalizes_once():
    # A caller can drain the body off the http response without ever calling
    # parse(). Closing it must finalize the span exactly once (so it does not
    # leak) and still run the real close.
    raw = _RawResponse(_FakeStream())
    finalize_calls = []
    # The proxy installs the close fallback in __init__; the closure it hangs on
    # http_response keeps it alive, so we drive the response directly.
    RawResponseStreamProxy(
        raw,
        wrap_stream=_SyncWrapper,
        finalize=lambda: finalize_calls.append(True),
    )

    raw.http_response.close()
    assert raw.http_response.close_calls == 1  # real close still ran
    assert finalize_calls == [True]  # span finalized

    raw.http_response.close()  # a second close must not finalize again
    assert finalize_calls == [True]


class _AsyncWrapper:
    """Stands in for an async stream wrapper, whose close() is a coroutine."""

    def __init__(self, stream=None):
        self.stream = stream
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1


def test_close_after_parse_drives_the_wrapper():
    # A wrapper owns finalization, but only once something drives it. A caller
    # that parses and then walks away (an early break, an exception inside the
    # with block) never does, so the close fallback closes the wrapper rather
    # than standing aside; the wrapper finalizes with what it saw.
    wrapper = _SyncWrapper()
    raw = _RawResponse(_FakeStream())
    finalize_calls = []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=lambda s: wrapper,
        finalize=lambda: finalize_calls.append(True),
    )

    proxy.parse()
    raw.http_response.close()
    assert raw.http_response.close_calls == 1  # real close still ran
    assert wrapper.close_calls == 1  # wrapper was driven
    assert finalize_calls == []  # the wrapper ends the span, not the fallback

    raw.http_response.close()  # a second close must not drive it again
    assert wrapper.close_calls == 1
    assert finalize_calls == []


def test_closing_the_wrapper_reentrantly_does_not_end_the_span():
    # Closing the wrapper closes the SDK stream, which closes this same httpx
    # response and re-enters the hook. The re-entrant call must not take the
    # "nobody parsed" branch and end the span before the wrapper finalizes it.
    raw = _RawResponse(_FakeStream())
    finalize_calls = []

    class _ReentrantWrapper:
        def close(self):
            raw.http_response.close()

    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=lambda s: _ReentrantWrapper(),
        finalize=lambda: finalize_calls.append(True),
    )

    proxy.parse()
    raw.http_response.close()
    assert finalize_calls == []  # the re-entrant close did not finalize


@pytest.mark.asyncio()
async def test_async_wrapper_is_closed_by_aclose_not_close():
    # httpx rejects a sync close of an async response and nothing in the sync
    # hook can await, so an async wrapper has to wait for the aclose hook.
    wrapper = _AsyncWrapper()
    raw = _AsyncRawResponse(_FakeAsyncStream())
    finalize_calls = []
    proxy = RawResponseStreamProxy(
        raw,
        wrap_stream=lambda s: wrapper,
        finalize=lambda: finalize_calls.append(True),
    )

    await proxy.parse()

    raw.http_response.close()  # sync hook leaves the wrapper alone
    assert wrapper.close_calls == 0
    assert finalize_calls == []

    await raw.http_response.aclose()
    assert wrapper.close_calls == 1  # aclose drove it
    assert finalize_calls == []


def test_wrap_result_non_stream_finalizes_on_close(
    tracer_provider, meter_provider, logger_provider, span_exporter
):
    # End-to-end: a parse result we can't wrap must not leak the span the
    # instrumentation already started. Drive the real wrap_stream_result path
    # and assert parse() leaves the span open, then closing the body finalizes
    # it.
    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    invocation = handler.inference("openai", request_model="gpt-4")

    raw = _RawResponse("not-a-stream")
    result = wrap_stream_result(
        ChatStreamWrapper, raw, invocation, capture_content=False
    )

    assert span_exporter.get_finished_spans() == ()  # not closed yet

    parsed = result.parse()  # non-stream handed back untouched
    assert parsed == "not-a-stream"
    assert span_exporter.get_finished_spans() == ()  # parse did not finalize

    raw.http_response.close()  # close fallback finalizes the span

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].end_time is not None
