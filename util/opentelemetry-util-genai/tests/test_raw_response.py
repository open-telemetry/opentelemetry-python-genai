# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared ``with_raw_response`` proxy state machine."""

import asyncio

import pytest

from opentelemetry.util.genai.raw_response import (
    RawResponseProxy,
    is_raw_response,
    wrap_raw_response,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)

_PAYLOAD = {"object": "payload", "id": "id-1"}


class _FakeInvocation:
    def __init__(self):
        self.stop_count = 0
        self.failures = []
        self._request_stream = False

    def stop(self):
        self.stop_count += 1

    def fail(self, error):
        self.failures.append(error)

    def _on_stream_chunk(self, chunk_at):
        pass


class _FakeHttpResponse:
    def __init__(self, body=_PAYLOAD, is_closed=False, content_type=None):
        self.body = body
        self.is_closed = is_closed
        self.headers = {"content-type": content_type or "application/json"}
        self.close_calls = 0
        self.aclose_calls = 0
        self.read_calls = 0
        self._read = False

    def json(self):
        if self.body is None:
            raise ValueError("not json")
        return self.body

    @property
    def content(self):
        if not self._read:
            raise RuntimeError("response not read")
        return b"{}"

    def read(self):
        self.read_calls += 1
        self._read = True
        return b"{}"

    async def aread(self):
        return self.read()

    def close(self):
        self.close_calls += 1

    async def aclose(self):
        self.aclose_calls += 1


class _FakeRaw:
    def __init__(self, parse_result=None, http_response=None, coro=False):
        self.http_response = (
            _FakeHttpResponse() if http_response is None else http_response
        )
        self._parse_result = parse_result
        self._coro = coro
        self.parse_calls = []

    def parse(self, *args, **kwargs):
        self.parse_calls.append((args, kwargs))
        if kwargs.get("to") is not None:
            result = kwargs["to"]
        else:
            result = self._parse_result
        if self._coro:

            async def _await():
                return result

            return _await()
        return result


class _FakeSyncStream:
    def __init__(self):
        self.close_calls = 0

    def __iter__(self):
        return iter(())

    def close(self):
        self.close_calls += 1


class _FakeAsyncStream:
    def __init__(self):
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        self.close_calls += 1


class _SyncWrapper(SyncStreamWrapper):
    def _process_chunk(self, chunk):
        pass

    def _on_stream_end(self):
        self._self_invocation.stop()

    def _on_stream_error(self, error):
        self._self_invocation.fail(error)


class _AsyncWrapper(AsyncStreamWrapper):
    def _process_chunk(self, chunk):
        pass

    def _on_stream_end(self):
        self._self_invocation.stop()

    def _on_stream_error(self, error):
        self._self_invocation.fail(error)


class _Proxy(RawResponseProxy):
    """Minimal subclass: recognizes ``_PAYLOAD`` and the two fake streams."""

    def __init__(self, raw, invocation, wrap_error=None):
        super().__init__(raw, invocation)
        self._self_recorded = []
        self._self_wrap_error = wrap_error

    def _extract_from_body(self, body):
        if body.get("object") != "payload":
            return False
        self._self_recorded.append(body)
        return True

    def _extract_from_parsed(self, parsed):
        if parsed is not _PAYLOAD:
            return False
        self._self_recorded.append(parsed)
        return True

    def _wrap_parsed_stream(self, parsed):
        if self._self_wrap_error is not None:
            raise self._self_wrap_error
        if isinstance(parsed, _FakeSyncStream):
            return _SyncWrapper(parsed, invocation=self._self_invocation)
        if isinstance(parsed, _FakeAsyncStream):
            return _AsyncWrapper(parsed, invocation=self._self_invocation)
        return None


def _wrap(raw, invocation, streamed=False, **kwargs):
    return wrap_raw_response(
        lambda r: _Proxy(r, invocation, **kwargs), raw, streamed=streamed
    )


def test_is_raw_response_matches_on_shape():
    assert is_raw_response(_FakeRaw())
    assert not is_raw_response(object())
    assert not is_raw_response(_FakeHttpResponse())


def test_closed_payload_body_is_recorded_and_not_proxied():
    invocation = _FakeInvocation()
    raw = _FakeRaw(http_response=_FakeHttpResponse(is_closed=True))

    result = _wrap(raw, invocation)

    assert result is raw  # nothing to defer to, so no proxy
    assert raw.parse_calls == []  # the caller's parse is never run for us
    assert invocation.stop_count == 1


def test_closed_unrecognized_body_still_finalizes():
    """A closed body we cannot read must not leave the invocation open.

    Nothing will read or close the response again, so this is the last chance
    to end it.
    """
    invocation = _FakeInvocation()
    raw = _FakeRaw(
        http_response=_FakeHttpResponse(
            body={"object": "other"}, is_closed=True
        )
    )

    _wrap(raw, invocation)

    assert invocation.stop_count == 1


def test_closed_non_json_body_still_finalizes():
    invocation = _FakeInvocation()
    raw = _FakeRaw(http_response=_FakeHttpResponse(body=None, is_closed=True))

    _wrap(raw, invocation)

    assert invocation.stop_count == 1


def test_closed_streamed_body_is_proxied():
    """A buffered transport can close a body that is still an unparsed stream.

    The caller told us the request asked for a stream, so ``is_closed`` does
    not mean the operation is over.
    """
    invocation = _FakeInvocation()
    stream = _FakeSyncStream()
    raw = _FakeRaw(stream, http_response=_FakeHttpResponse(is_closed=True))

    proxy = _wrap(raw, invocation, streamed=True)

    assert proxy is not raw
    assert invocation.stop_count == 0
    assert isinstance(proxy.parse(), _SyncWrapper)


def test_response_without_http_response_is_proxied_without_hooks():
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD)
    raw.http_response = None

    proxy = _wrap(raw, invocation)

    assert proxy.parse() is _PAYLOAD
    assert invocation.stop_count == 1


def test_parse_records_payload_and_finalizes_once():
    invocation = _FakeInvocation()
    proxy = _wrap(_FakeRaw(_PAYLOAD), invocation)

    assert proxy.parse() is _PAYLOAD
    assert invocation.stop_count == 1

    # A second parse must not record or finalize again.
    assert proxy.parse() is _PAYLOAD
    assert invocation.stop_count == 1


def test_parse_stream_hands_span_to_the_wrapper():
    invocation = _FakeInvocation()
    stream = _FakeSyncStream()
    proxy = _wrap(_FakeRaw(stream), invocation)

    wrapper = proxy.parse()

    assert isinstance(wrapper, _SyncWrapper)
    assert invocation.stop_count == 0  # the wrapper owns it now
    assert (
        proxy.parse() is wrapper
    )  # replayed, never re-parsed into a raw stream

    wrapper.close()
    assert invocation.stop_count == 1


def test_abandoned_stream_is_closed_by_the_close_hook():
    invocation = _FakeInvocation()
    stream = _FakeSyncStream()
    raw = _FakeRaw(stream)
    proxy = _wrap(raw, invocation)

    proxy.parse()
    raw.http_response.close()

    assert stream.close_calls == 1
    assert invocation.stop_count == 1


def test_sync_close_of_an_async_stream_defers_to_aclose():
    async def exercise():
        invocation = _FakeInvocation()
        stream = _FakeAsyncStream()
        raw = _FakeRaw(stream)
        proxy = _wrap(raw, invocation)

        proxy.parse()
        # Nothing here can await, so the sync close must leave the wrapper for
        # aclose rather than dropping it.
        raw.http_response.close()
        assert invocation.stop_count == 0

        await raw.http_response.aclose()
        assert stream.close_calls == 1
        assert invocation.stop_count == 1

    asyncio.run(exercise())


def test_unparseable_value_is_returned_untouched_and_finalizes():
    invocation = _FakeInvocation()
    proxy = _wrap(_FakeRaw("not-a-payload"), invocation)

    assert proxy.parse() == "not-a-payload"
    assert invocation.stop_count == 1


def test_wrap_failure_does_not_break_the_caller():
    invocation = _FakeInvocation()
    proxy = _wrap(
        _FakeRaw(_FakeSyncStream()),
        invocation,
        wrap_error=ValueError("boom"),
    )

    parsed = proxy.parse()

    assert isinstance(parsed, _FakeSyncStream)  # handed back uninstrumented
    assert invocation.stop_count == 1


def test_cast_parse_wrap_failure_does_not_break_the_caller():
    invocation = _FakeInvocation()
    raw = _FakeRaw(None)
    proxy = _wrap(raw, invocation, wrap_error=ValueError("boom"))

    stream = _FakeSyncStream()
    assert proxy.parse(to=stream) is stream


def test_cast_parse_is_delegated():
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD)
    proxy = _wrap(raw, invocation)

    sentinel = object()
    assert proxy.parse(to=sentinel) is sentinel
    assert raw.parse_calls == [((), {"to": sentinel})]


def test_cast_parse_of_a_stream_is_wrapped():
    invocation = _FakeInvocation()
    proxy = _wrap(_FakeRaw(None), invocation)

    stream = _FakeSyncStream()
    wrapper = proxy.parse(to=stream)

    assert isinstance(wrapper, _SyncWrapper)
    assert invocation.stop_count == 0


def test_read_without_parse_records_the_body():
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD)
    proxy = _wrap(raw, invocation)

    raw.http_response.read()

    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1
    assert raw.parse_calls == []


def test_close_without_parse_finalizes_once():
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD)
    _wrap(raw, invocation)

    raw.http_response.close()
    raw.http_response.close()

    assert invocation.stop_count == 1


def test_parse_after_a_body_read_does_not_record_twice():
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD)
    proxy = _wrap(raw, invocation)

    raw.http_response.read()
    assert proxy.parse() is _PAYLOAD

    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1


def test_failing_parse_propagates_and_finalizes():
    invocation = _FakeInvocation()

    class _Raw(_FakeRaw):
        def parse(self, *args, **kwargs):
            raise ValueError("caller parse boom")

    proxy = _wrap(_Raw(), invocation)

    with pytest.raises(ValueError, match="caller parse boom"):
        proxy.parse()

    assert invocation.stop_count == 1


def test_hooks_stack_over_one_response():
    """Two proxies over one response each finalize their own invocation."""
    first_invocation = _FakeInvocation()
    second_invocation = _FakeInvocation()
    http_response = _FakeHttpResponse()
    first_raw = _FakeRaw(_PAYLOAD, http_response=http_response)
    second_raw = _FakeRaw(_PAYLOAD, http_response=http_response)

    _wrap(first_raw, first_invocation)
    _wrap(second_raw, second_invocation)

    http_response.close()

    assert first_invocation.stop_count == 1
    assert second_invocation.stop_count == 1
    assert http_response.close_calls == 1


def test_async_parse_coroutine_is_awaited_and_dispatched():
    async def exercise():
        invocation = _FakeInvocation()
        proxy = _wrap(_FakeRaw(_PAYLOAD, coro=True), invocation)

        assert await proxy.parse() is _PAYLOAD
        assert invocation.stop_count == 1

    asyncio.run(exercise())


def test_async_parse_failure_propagates_and_finalizes():
    async def exercise():
        invocation = _FakeInvocation()

        class _Raw(_FakeRaw):
            def parse(self, *args, **kwargs):
                async def _boom():
                    raise ValueError("async parse boom")

                return _boom()

        proxy = _wrap(_Raw(), invocation)

        with pytest.raises(ValueError, match="async parse boom"):
            await proxy.parse()

        assert invocation.stop_count == 1

    asyncio.run(exercise())


def test_unawaited_async_parse_does_not_strand_the_read_hook():
    """Building the coroutine reads nothing, so the read hook stays armed."""
    invocation = _FakeInvocation()
    raw = _FakeRaw(_PAYLOAD, coro=True)
    proxy = _wrap(raw, invocation)

    coro = proxy.parse()
    coro.close()  # the caller never awaited it

    raw.http_response.read()
    assert invocation.stop_count == 1


def test_cast_parse_does_not_poison_the_default_parse_memo():
    """``parse(to=X)`` must not make a later ``parse()`` return X's wrapper."""
    invocation = _FakeInvocation()
    default_stream = _FakeSyncStream()
    proxy = _wrap(_FakeRaw(default_stream), invocation)

    cast_stream = _FakeSyncStream()
    cast_wrapper = proxy.parse(to=cast_stream)
    assert cast_wrapper.__wrapped__ is cast_stream

    # The default parse must never be served the cast target's wrapper. The
    # body is spent by now, so it gets the SDK's own object uninstrumented.
    assert proxy.parse() is default_stream


def test_cast_parsed_stream_is_finalized_by_the_close_hook():
    invocation = _FakeInvocation()
    raw = _FakeRaw(None)
    proxy = _wrap(raw, invocation)

    cast_stream = _FakeSyncStream()
    proxy.parse(to=cast_stream)
    raw.http_response.close()

    assert cast_stream.close_calls == 1
    assert invocation.stop_count == 1


class _AsyncParseRaw(_FakeRaw):
    """A client whose ``parse`` is a real coroutine function."""

    async def parse(self, *args, **kwargs):
        self.parse_calls.append((args, kwargs))
        if kwargs.get("to") is not None:
            return kwargs["to"]
        return self._parse_result


def test_async_client_replays_the_wrapper_as_a_coroutine():
    async def exercise():
        invocation = _FakeInvocation()
        proxy = _wrap(_AsyncParseRaw(_FakeAsyncStream()), invocation)

        wrapper = await proxy.parse()
        assert isinstance(wrapper, _AsyncWrapper)
        # The replay has to stay awaitable for an async client.
        assert await proxy.parse() is wrapper

    asyncio.run(exercise())


def test_async_abandoned_stream_is_closed_by_the_async_close_hook():
    async def exercise():
        invocation = _FakeInvocation()
        stream = _FakeAsyncStream()
        raw = _AsyncParseRaw(stream)
        proxy = _wrap(raw, invocation)

        await proxy.parse()
        await raw.http_response.aclose()

        assert stream.close_calls == 1
        assert invocation.stop_count == 1

    asyncio.run(exercise())


class _NestedReadHttpResponse(_FakeHttpResponse):
    """A transport whose ``read`` re-enters itself once.

    Contrived, but it isolates the depth behaviour: with a plain in-flight
    flag the inner read clears it and the outer read finalizes early, before
    the body it is still draining is available.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.depth = 0
        self.finalized_at_depth = []

    def read(self):
        self.depth += 1
        try:
            if self.depth == 1:
                # The re-entrant read goes through the installed hook too.
                self.read()
            return super().read()
        finally:
            self.depth -= 1


def test_nested_read_finalizes_only_once_the_outer_read_returns():
    invocation = _FakeInvocation()
    http_response = _NestedReadHttpResponse()
    raw = _FakeRaw(_PAYLOAD, http_response=http_response)
    proxy = _wrap(raw, invocation)

    http_response.read()

    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1


def test_read_driven_by_parse_leaves_finalization_to_the_dispatch():
    """The read hook must stand down for the read ``parse()`` drives."""
    invocation = _FakeInvocation()

    class _ReadingRaw(_FakeRaw):
        def parse(self, *args, **kwargs):
            self.http_response.read()  # the SDK reads the body here
            return super().parse(*args, **kwargs)

    raw = _ReadingRaw(_PAYLOAD)
    proxy = _wrap(raw, invocation)

    assert proxy.parse() is _PAYLOAD

    # Recorded once, by the parse dispatch rather than twice via the read hook.
    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1


def test_close_during_a_parse_driven_read_stands_down():
    """httpx closes while draining, before the content is available."""
    invocation = _FakeInvocation()

    class _ClosingRaw(_FakeRaw):
        def parse(self, *args, **kwargs):
            self.http_response.close()  # closed mid-drain
            return super().parse(*args, **kwargs)

    raw = _ClosingRaw(_PAYLOAD)
    proxy = _wrap(raw, invocation)

    assert proxy.parse() is _PAYLOAD
    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1


def test_close_during_a_plain_read_stands_down():
    invocation = _FakeInvocation()

    class _ClosingHttpResponse(_FakeHttpResponse):
        def read(self):
            self.close()  # httpx closes before assigning the content
            return super().read()

    http_response = _ClosingHttpResponse()
    raw = _FakeRaw(_PAYLOAD, http_response=http_response)
    proxy = _wrap(raw, invocation)

    http_response.read()

    assert proxy._self_recorded == [_PAYLOAD]
    assert invocation.stop_count == 1


def test_reads_in_flight_returns_to_zero_after_a_failed_parse():
    invocation = _FakeInvocation()

    class _Raw(_FakeRaw):
        def parse(self, *args, **kwargs):
            raise ValueError("boom")

    proxy = _wrap(_Raw(), invocation)

    with pytest.raises(ValueError, match="boom"):
        proxy.parse()

    assert proxy._self_reads_in_flight == 0
