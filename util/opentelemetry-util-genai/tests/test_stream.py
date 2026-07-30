# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=abstract-class-instantiated

import asyncio
import inspect
import timeit
from unittest.mock import patch

import pytest

from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)


def test_stream_wrapper_abstract_method_signatures_match():
    method_names = (
        "_process_chunk",
        "_on_stream_end",
        "_on_stream_error",
    )

    for method_name in method_names:
        assert inspect.signature(
            getattr(SyncStreamWrapper, method_name)
        ) == inspect.signature(getattr(AsyncStreamWrapper, method_name))


class _FakeSyncStream:
    def __init__(self, chunks=None, error=None, close_error=None):
        self._chunks = list(chunks or [])
        self._error = error
        self._close_error = close_error
        self.close_count = 0
        self.extra_attribute = "passthrough"

    def __iter__(self):
        return self

    def __next__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopIteration

    def close(self):
        self.close_count += 1
        if self._close_error:
            raise self._close_error

    def __len__(self):
        return 42


class _FakeSyncIterable:
    def __init__(self, chunks=None):
        self.iterator = iter(chunks or [])
        self.close_count = 0

    def __iter__(self):
        return self.iterator

    def close(self):
        self.close_count += 1


class _TestSyncStreamWrapper(SyncStreamWrapper):
    def __init__(self, stream):
        super().__init__(stream)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1

    def _on_stream_error(self, error):
        self._self_failures.append(error)


class _FailingSyncProcessStreamWrapper(_TestSyncStreamWrapper):
    def _process_chunk(self, chunk):
        raise ValueError("instrumentation failed")


class _FailingSyncStopStreamWrapper(_TestSyncStreamWrapper):
    def _on_stream_end(self):
        self._self_stop_count += 1
        raise ValueError("instrumentation failed")


class _FailingSyncFailStreamWrapper(_TestSyncStreamWrapper):
    def _on_stream_error(self, error):
        self._self_failures.append(error)
        raise ValueError("instrumentation failed")


def test_sync_stream_wrapper_processes_chunks_and_stops():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    assert next(wrapper) == "chunk"
    assert wrapper._self_processed == ["chunk"]

    try:
        next(wrapper)
    except StopIteration:
        pass

    assert wrapper._self_stop_count == 1


def test_sync_stream_wrapper_processes_iterables():
    stream = _FakeSyncIterable(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    assert next(wrapper) == "chunk"
    assert wrapper._self_processed == ["chunk"]

    with pytest.raises(StopIteration):
        next(wrapper)

    assert wrapper._self_stop_count == 1


def test_sync_stream_wrapper_fails_stream_errors():
    error = ValueError("boom")
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream(error=error))

    try:
        next(wrapper)
    except ValueError:
        pass

    assert wrapper._self_failures == [error]


def test_sync_stream_wrapper_close_stops_once():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)

    wrapper.close()
    wrapper.close()

    assert stream.close_count == 2
    assert wrapper._self_stop_count == 1
    assert not wrapper._self_failures


def test_sync_stream_wrapper_close_fails_with_close_error():
    error = RuntimeError("close failure")
    wrapper = _TestSyncStreamWrapper(
        _FakeSyncStream(chunks=["chunk"], close_error=error)
    )

    with pytest.raises(RuntimeError, match="close failure"):
        wrapper.close()

    assert wrapper._self_failures == [error]
    assert wrapper._self_stop_count == 0


def test_sync_stream_wrapper_exit_closes_and_propagates_user_errors():
    stream = _FakeSyncStream(chunks=["chunk"])
    wrapper = _TestSyncStreamWrapper(stream)
    error = RuntimeError("user failure")

    assert wrapper.__exit__(RuntimeError, error, None) is False

    assert stream.close_count == 1
    assert wrapper._self_stop_count == 0
    assert wrapper._self_failures == [error]


def test_sync_stream_wrapper_getattr_passthrough():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    assert wrapper.extra_attribute == "passthrough"


def test_sync_stream_wrapper_exposes_wrapped_stream():
    stream = _FakeSyncStream()
    wrapper = _TestSyncStreamWrapper(stream)

    assert getattr(wrapper, "__wrapped__") is stream


def test_sync_stream_wrapper_magic_method_passthrough():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    assert len(wrapper) == 42


def test_sync_stream_wrapper_stop_iteration_does_not_double_finalize():
    wrapper = _TestSyncStreamWrapper(_FakeSyncStream())

    with pytest.raises(StopIteration):
        next(wrapper)
    wrapper.close()

    assert wrapper._self_stop_count == 1
    assert not wrapper._self_failures


class _FakeAsyncStream:
    def __init__(self, chunks=None, error=None, close_error=None):
        self._chunks = list(chunks or [])
        self._error = error
        self._close_error = close_error
        self.close_count = 0
        self.extra_attribute = "passthrough"

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        if self._error:
            raise self._error
        raise StopAsyncIteration

    async def close(self):
        self.close_count += 1
        if self._close_error:
            raise self._close_error

    def __len__(self):
        return 42


class _FakeAsyncIterable:
    def __init__(self, chunks=None):
        self.iterator = _FakeAsyncStream(chunks=chunks)
        self.close_count = 0

    def __aiter__(self):
        return self.iterator

    async def close(self):
        self.close_count += 1


class _TestAsyncStreamWrapper(AsyncStreamWrapper):
    def __init__(self, stream):
        super().__init__(stream)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1

    def _on_stream_error(self, error):
        self._self_failures.append(error)


def test_async_stream_wrapper_processes_chunks_and_stops():
    async def exercise():
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream(chunks=["chunk"]))

        assert await anext(wrapper) == "chunk"
        assert wrapper._self_processed == ["chunk"]

        try:
            await anext(wrapper)
        except StopAsyncIteration:
            pass

        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_processes_async_iterables():
    async def exercise():
        stream = _FakeAsyncIterable(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)

        assert await anext(wrapper) == "chunk"
        assert wrapper._self_processed == ["chunk"]

        with pytest.raises(StopAsyncIteration):
            await anext(wrapper)

        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_fails_stream_errors():
    async def exercise():
        error = ValueError("boom")
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream(error=error))

        with pytest.raises(ValueError):
            await anext(wrapper)

        assert wrapper._self_failures == [error]

    asyncio.run(exercise())


def test_async_stream_wrapper_close_stops_once():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)

        await wrapper.close()
        await wrapper.close()

        assert stream.close_count == 2
        assert wrapper._self_stop_count == 1
        assert not wrapper._self_failures

    asyncio.run(exercise())


def test_async_stream_wrapper_close_fails_with_close_error():
    async def exercise():
        error = RuntimeError("close failure")
        wrapper = _TestAsyncStreamWrapper(
            _FakeAsyncStream(chunks=["chunk"], close_error=error)
        )

        with pytest.raises(RuntimeError, match="close failure"):
            await wrapper.close()

        assert wrapper._self_failures == [error]
        assert wrapper._self_stop_count == 0

    asyncio.run(exercise())


def test_async_stream_wrapper_exit_closes_and_propagates_user_errors():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["chunk"])
        wrapper = _TestAsyncStreamWrapper(stream)
        error = RuntimeError("user failure")

        assert await wrapper.__aexit__(RuntimeError, error, None) is False

        assert stream.close_count == 1
        assert wrapper._self_stop_count == 0
        assert wrapper._self_failures == [error]

    asyncio.run(exercise())


def test_async_stream_wrapper_getattr_passthrough():
    wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

    assert wrapper.extra_attribute == "passthrough"


def test_async_stream_wrapper_exposes_wrapped_stream():
    stream = _FakeAsyncStream()
    wrapper = _TestAsyncStreamWrapper(stream)

    assert getattr(wrapper, "__wrapped__") is stream


def test_async_stream_wrapper_magic_method_passthrough():
    wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

    assert len(wrapper) == 42


def test_async_stream_wrapper_stop_iteration_does_not_double_finalize():
    async def exercise():
        wrapper = _TestAsyncStreamWrapper(_FakeAsyncStream())

        with pytest.raises(StopAsyncIteration):
            await anext(wrapper)
        await wrapper.close()

        assert wrapper._self_stop_count == 1
        assert not wrapper._self_failures

    asyncio.run(exercise())


# --- Streaming timing seam tests ---
#
# The wrapper does not compute TTFC/gaps itself: when an invocation is passed
# it reports each chunk's arrival time via invocation._on_stream_chunk, and the
# invocation turns those timestamps into metrics (covered by
# test_handler_metrics). These tests cover the wrapper's side of that seam.


class _FakeTimingInvocation:
    """Minimal stand-in for InferenceInvocation's timing seam."""

    def __init__(self):
        self.chunk_times = []
        self._request_stream = None

    def _on_stream_chunk(self, chunk_at):
        self.chunk_times.append(chunk_at)


class _TimingSyncWrapper(SyncStreamWrapper):
    def __init__(self, stream, invocation=None, process_hook=None):
        super().__init__(stream, invocation=invocation)
        self._self_processed = []
        self._self_process_hook = process_hook

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)
        if self._self_process_hook is not None:
            self._self_process_hook()

    def _on_stream_end(self):
        pass

    def _on_stream_error(self, error):
        pass


class _TimingAsyncWrapper(AsyncStreamWrapper):
    def __init__(self, stream, invocation=None):
        super().__init__(stream, invocation=invocation)
        self._self_processed = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        pass

    def _on_stream_error(self, error):
        pass


def test_sync_wrapper_reports_each_chunk_arrival():
    invocation = _FakeTimingInvocation()
    stream = _FakeSyncStream(chunks=["a", "b", "c"])
    wrapper = _TimingSyncWrapper(stream, invocation=invocation)

    with patch(
        "timeit.default_timer", side_effect=iter([101.2, 101.8, 102.1])
    ):
        assert list(wrapper) == ["a", "b", "c"]

    assert invocation.chunk_times == pytest.approx([101.2, 101.8, 102.1])


def test_sync_wrapper_marks_request_stream():
    invocation = _FakeTimingInvocation()
    # The wrapper marks the request as streamed at construction, before any
    # chunk is read.
    _TimingSyncWrapper(_FakeSyncStream(chunks=["a"]), invocation=invocation)
    assert invocation._request_stream is True


def test_sync_wrapper_single_chunk_one_report():
    invocation = _FakeTimingInvocation()
    stream = _FakeSyncStream(chunks=["only"])
    wrapper = _TimingSyncWrapper(stream, invocation=invocation)

    with patch("timeit.default_timer", side_effect=iter([60.5])):
        assert list(wrapper) == ["only"]

    assert invocation.chunk_times == pytest.approx([60.5])


def test_sync_wrapper_without_invocation_skips_timing():
    stream = _FakeSyncStream(chunks=["a", "b"])
    wrapper = _TimingSyncWrapper(stream)

    with patch("timeit.default_timer") as timer:
        assert list(wrapper) == ["a", "b"]

    timer.assert_not_called()


def test_sync_wrapper_error_before_first_chunk_no_report():
    invocation = _FakeTimingInvocation()
    stream = _FakeSyncStream(error=RuntimeError("network"))
    wrapper = _TimingSyncWrapper(stream, invocation=invocation)

    with pytest.raises(RuntimeError, match="network"):
        next(wrapper)

    assert invocation.chunk_times == []


def test_sync_wrapper_captures_arrival_before_processing():
    """Arrival time is taken before _process_chunk, so per-chunk timing
    excludes the instrumentation's own processing of the chunk."""
    invocation = _FakeTimingInvocation()
    stream = _FakeSyncStream(chunks=["a"])
    # First clock read is the chunk arrival (100.0); the second is consumed
    # inside _process_chunk to simulate processing taking time.
    times = iter([100.0, 100.9])
    wrapper = _TimingSyncWrapper(
        stream,
        invocation=invocation,
        process_hook=lambda: timeit.default_timer(),
    )

    with patch("timeit.default_timer", side_effect=times):
        list(wrapper)

    assert invocation.chunk_times == pytest.approx([100.0])


def test_async_wrapper_reports_each_chunk_arrival():
    async def exercise():
        invocation = _FakeTimingInvocation()
        stream = _FakeAsyncStream(chunks=["x", "y", "z"])
        wrapper = _TimingAsyncWrapper(stream, invocation=invocation)

        with patch(
            "timeit.default_timer", side_effect=iter([201.3, 202.0, 202.2])
        ):
            chunks = [chunk async for chunk in wrapper]

        assert chunks == ["x", "y", "z"]
        assert invocation.chunk_times == pytest.approx([201.3, 202.0, 202.2])

    asyncio.run(exercise())


def test_async_wrapper_without_invocation_skips_timing():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["a", "b"])
        wrapper = _TimingAsyncWrapper(stream)

        with patch("timeit.default_timer") as timer:
            chunks = [chunk async for chunk in wrapper]

        assert chunks == ["a", "b"]
        timer.assert_not_called()

    asyncio.run(exercise())
