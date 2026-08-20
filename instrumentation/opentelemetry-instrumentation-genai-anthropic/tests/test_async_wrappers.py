# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.genai.anthropic.wrappers import (
    AsyncMessagesStreamManagerWrapper,
    AsyncMessagesStreamWrapper,
    MessagesStreamManagerWrapper,
    MessagesStreamWrapper,
)


def _make_invocation():
    return SimpleNamespace(
        attributes={},
        request_model=None,
        stop=lambda: None,
        fail=lambda error: None,
        _on_stream_chunk=lambda chunk_at: None,
    )


def _make_stream_wrapper(stream):
    return MessagesStreamWrapper(
        stream=stream,
        invocation=_make_invocation(),
        capture_content=False,
    )


def _make_async_stream_wrapper(stream):
    return AsyncMessagesStreamWrapper(
        stream=stream,
        invocation=_make_invocation(),
        capture_content=False,
    )


class _FakeSyncStream:
    def __init__(
        self,
        *,
        events=None,
        error=None,
        close_error=None,
        current_message_snapshot=None,
    ):
        self._events = list(events or [])
        self._error = error
        self._close_error = close_error
        self.current_message_snapshot = current_message_snapshot
        self.close_calls = 0
        self.response = _FakeSyncResponse()

    def __iter__(self):
        return self

    def __next__(self):
        if self._events:
            return self._events.pop(0)
        if self._error is not None:
            raise self._error
        raise StopIteration

    def close(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class _FakeAsyncStream:
    def __init__(
        self,
        *,
        events=None,
        error=None,
        close_error=None,
        current_message_snapshot=None,
    ):
        self._events = list(events or [])
        self._error = error
        self._close_error = close_error
        self.current_message_snapshot = current_message_snapshot
        self.close_calls = 0
        self.final_message = SimpleNamespace(id="msg_final")
        self.response = _FakeAsyncResponse()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._events:
            return self._events.pop(0)
        if self._error is not None:
            raise self._error
        raise StopAsyncIteration

    async def close(self):
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error

    async def get_final_message(self):
        return self.final_message


class _FakeSyncManager:
    def __init__(
        self, stream, suppressed=False, enter_error=None, exit_error=None
    ):
        self._stream = stream
        self._suppressed = suppressed
        self._enter_error = enter_error
        self._exit_error = exit_error
        self.exit_args = None

    def __enter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        if self._exit_error is not None:
            raise self._exit_error
        return self._suppressed


class _FakeAsyncManager:
    def __init__(
        self, stream, suppressed=False, enter_error=None, exit_error=None
    ):
        self._stream = stream
        self._suppressed = suppressed
        self._enter_error = enter_error
        self._exit_error = exit_error
        self.exit_args = None

    async def __aenter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        if self._exit_error is not None:
            raise self._exit_error
        return self._suppressed


class _FakeSyncResponse:
    def __init__(self):
        self.request_id = "req_sync"
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _FakeAsyncResponse:
    def __init__(self):
        self.request_id = "req_async"
        self.close_calls = 0
        self.aclose_calls = 0

    def close(self):
        self.close_calls += 1

    async def aclose(self):
        self.aclose_calls += 1


def test_sync_stream_wrapper_exit_closes_without_exception():
    stream = _FakeSyncStream()
    wrapper = _make_stream_wrapper(stream)
    stopped = []

    wrapper._stop = lambda: stopped.append(True)

    result = wrapper.__exit__(None, None, None)

    assert result is False
    assert stream.close_calls == 1
    assert stopped == [True]


def test_sync_stream_wrapper_exit_fails_and_closes_on_exception():
    stream = _FakeSyncStream()
    wrapper = _make_stream_wrapper(stream)
    stopped = []
    failures = []

    wrapper._stop = lambda: stopped.append(True)
    wrapper._fail = failures.append

    error = ValueError("boom")
    result = wrapper.__exit__(ValueError, error, None)

    assert result is False
    assert stream.close_calls == 1
    assert not stopped
    assert failures == [error]


def test_sync_stream_wrapper_close_failure_fails_and_reraises():
    error = RuntimeError("close failed")
    stream = _FakeSyncStream(close_error=error)
    wrapper = _make_stream_wrapper(stream)
    stopped = []
    failures = []

    wrapper._stop = lambda: stopped.append(True)
    wrapper._fail = failures.append

    with pytest.raises(RuntimeError, match="close failed"):
        wrapper.close()

    assert stream.close_calls == 1
    assert failures == [error]
    assert not stopped


def test_sync_stream_wrapper_processes_events_and_stops_on_completion():
    event = SimpleNamespace(type="message_start")
    stream = _FakeSyncStream(events=[event])
    wrapper = _make_stream_wrapper(stream)
    processed = []
    stopped = []

    wrapper._process_chunk = processed.append
    wrapper._stop = lambda: stopped.append(True)

    result = next(wrapper)

    assert result is event
    assert processed == [event]

    with pytest.raises(StopIteration):
        next(wrapper)

    assert stopped == [True]


def test_sync_stream_wrapper_uses_current_message_snapshot():
    event = SimpleNamespace(type="message_delta")
    snapshot = SimpleNamespace(id="msg_sync")
    stream = _FakeSyncStream(
        events=[event],
        current_message_snapshot=snapshot,
    )
    wrapper = _make_stream_wrapper(stream)

    result = next(wrapper)

    assert result is event
    assert wrapper._self_message is snapshot


def test_sync_stream_wrapper_fails_and_reraises_stream_errors():
    error = ValueError("boom")
    stream = _FakeSyncStream(error=error)
    wrapper = _make_stream_wrapper(stream)
    failures = []

    wrapper._fail = failures.append

    with pytest.raises(ValueError, match="boom"):
        next(wrapper)

    assert failures == [error]


def test_sync_stream_wrapper_getattr_passthrough():
    stream = _FakeSyncStream()
    wrapper = _make_stream_wrapper(stream)

    assert wrapper.response.request_id == "req_sync"


def test_sync_stream_response_close_finalizes_wrapper():
    stream = _FakeSyncStream()
    wrapper = _make_stream_wrapper(stream)
    stopped = []

    wrapper._stop = lambda: stopped.append(True)

    wrapper.response.close()

    assert stream.response.close_calls == 1
    assert stopped == [True]


def test_sync_manager_enter_constructs_stream_wrapper():
    stream = _FakeSyncStream()
    wrapper = MessagesStreamManagerWrapper(
        manager=_FakeSyncManager(stream=stream),
        invocation_factory=_make_invocation,
        capture_content=False,
    )

    with wrapper as result:
        assert isinstance(result, MessagesStreamWrapper)
        assert result.stream is stream


def test_sync_manager_does_not_create_invocation_until_enter():
    stream = _FakeSyncStream()
    factory_calls = []
    wrapper = MessagesStreamManagerWrapper(
        manager=_FakeSyncManager(stream=stream),
        invocation_factory=lambda: (
            factory_calls.append(True) or _make_invocation()
        ),
        capture_content=False,
    )

    assert not factory_calls

    with wrapper:
        pass

    assert factory_calls == [True]


@pytest.mark.asyncio
async def test_async_stream_wrapper_exit_closes_without_exception():
    stream = _FakeAsyncStream()
    wrapper = _make_async_stream_wrapper(stream)
    stopped = []

    wrapper._stop = lambda: stopped.append(True)

    result = await wrapper.__aexit__(None, None, None)

    assert result is False
    assert stream.close_calls == 1
    assert stopped == [True]


@pytest.mark.asyncio
async def test_async_stream_wrapper_exit_fails_and_closes_on_exception():
    stream = _FakeAsyncStream()
    wrapper = _make_async_stream_wrapper(stream)
    stopped = []
    failures = []

    wrapper._stop = lambda: stopped.append(True)
    wrapper._fail = failures.append

    error = ValueError("boom")
    result = await wrapper.__aexit__(ValueError, error, None)

    assert result is False
    assert stream.close_calls == 1
    assert not stopped
    assert failures == [error]


@pytest.mark.asyncio
async def test_async_stream_wrapper_close_uses_close_and_stops():
    stream = _FakeAsyncStream()
    wrapper = _make_async_stream_wrapper(stream)
    stopped = []

    wrapper._stop = lambda: stopped.append(True)

    await wrapper.close()

    assert stream.close_calls == 1
    assert stopped == [True]


@pytest.mark.asyncio
async def test_async_stream_wrapper_close_failure_fails_and_reraises():
    error = RuntimeError("close failed")
    stream = _FakeAsyncStream(close_error=error)
    wrapper = _make_async_stream_wrapper(stream)
    stopped = []
    failures = []

    wrapper._stop = lambda: stopped.append(True)
    wrapper._fail = failures.append

    with pytest.raises(RuntimeError, match="close failed"):
        await wrapper.close()

    assert stream.close_calls == 1
    assert failures == [error]
    assert not stopped


@pytest.mark.asyncio
async def test_async_stream_wrapper_processes_events_and_stops_on_completion():
    event = SimpleNamespace(type="message_start")
    stream = _FakeAsyncStream(events=[event])
    wrapper = _make_async_stream_wrapper(stream)
    processed = []
    stopped = []

    wrapper._process_chunk = processed.append
    wrapper._stop = lambda: stopped.append(True)

    result = await anext(wrapper)

    assert result is event
    assert processed == [event]

    with pytest.raises(StopAsyncIteration):
        await anext(wrapper)

    assert stopped == [True]


@pytest.mark.asyncio
async def test_async_stream_wrapper_uses_current_message_snapshot():
    event = SimpleNamespace(type="message_delta")
    snapshot = SimpleNamespace(id="msg_async")
    stream = _FakeAsyncStream(
        events=[event],
        current_message_snapshot=snapshot,
    )
    wrapper = _make_async_stream_wrapper(stream)

    result = await anext(wrapper)

    assert result is event
    assert wrapper._self_message is snapshot


@pytest.mark.asyncio
async def test_async_stream_wrapper_fails_and_reraises_stream_errors():
    error = ValueError("boom")
    stream = _FakeAsyncStream(error=error)
    wrapper = _make_async_stream_wrapper(stream)
    failures = []

    wrapper._fail = failures.append

    with pytest.raises(ValueError, match="boom"):
        await anext(wrapper)

    assert failures == [error]


@pytest.mark.asyncio
async def test_async_stream_wrapper_preserves_stream_helper_methods():
    stream = _FakeAsyncStream()
    wrapper = _make_async_stream_wrapper(stream)

    result = await wrapper.get_final_message()

    assert result is stream.final_message
    assert wrapper.response.request_id == "req_async"


@pytest.mark.asyncio
async def test_async_stream_response_aclose_finalizes_wrapper():
    stream = _FakeAsyncStream()
    wrapper = _make_async_stream_wrapper(stream)
    stopped = []

    wrapper._stop = lambda: stopped.append(True)

    await wrapper.response.aclose()

    assert stream.response.aclose_calls == 1
    assert stopped == [True]


@pytest.mark.asyncio
async def test_async_manager_enter_constructs_async_stream_wrapper():
    stream = _FakeAsyncStream()
    wrapper = AsyncMessagesStreamManagerWrapper(
        manager=_FakeAsyncManager(stream=stream),
        invocation_factory=_make_invocation,
        capture_content=False,
    )

    async with wrapper as result:
        assert isinstance(result, AsyncMessagesStreamWrapper)
        assert result.stream is stream
