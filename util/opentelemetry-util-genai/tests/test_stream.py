# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=abstract-class-instantiated

import asyncio
import inspect
import timeit
from unittest.mock import patch

import pytest

from opentelemetry.util.genai.stream import (
    AsyncStreamManagerWrapper,
    AsyncStreamWrapper,
    SyncStreamManagerWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
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
    def __init__(self, stream, invocation=None):
        super().__init__(stream, invocation=invocation)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1
        if self._self_invocation is not None:
            self._self_invocation.stop()

    def _on_stream_error(self, error):
        self._self_failures.append(error)
        if self._self_invocation is not None:
            self._self_invocation.fail(error)


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

    assert wrapper.__wrapped__ is stream


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
    def __init__(self, stream, invocation=None):
        super().__init__(stream, invocation=invocation)
        self._self_processed = []
        self._self_stop_count = 0
        self._self_failures = []

    def _process_chunk(self, chunk):
        self._self_processed.append(chunk)

    def _on_stream_end(self):
        self._self_stop_count += 1
        if self._self_invocation is not None:
            self._self_invocation.stop()

    def _on_stream_error(self, error):
        self._self_failures.append(error)
        if self._self_invocation is not None:
            self._self_invocation.fail(error)


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

    assert wrapper.__wrapped__ is stream


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


class _FakeAcloseOnlyStream:
    """An async generator style stream: ``aclose``, no ``close``."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks or [])
        self.aclose_count = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration

    async def aclose(self):
        self.aclose_count += 1


def test_async_stream_wrapper_aclose_finalizes_telemetry():
    async def exercise():
        stream = _FakeAcloseOnlyStream(chunks=["a"])
        wrapper = _TestAsyncStreamWrapper(stream)

        await wrapper.aclose()

        assert stream.aclose_count == 1
        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_stream_wrapper_only_exposes_the_streams_own_close_method():
    close_only = _TestAsyncStreamWrapper(_FakeAsyncStream(chunks=["a"]))
    aclose_only = _TestAsyncStreamWrapper(_FakeAcloseOnlyStream(chunks=["a"]))

    assert not hasattr(close_only, "aclose")
    assert not hasattr(aclose_only, "close")


class _FakeBothCloseStream(_FakeAcloseOnlyStream):
    """A stream pairing a sync ``close`` with an async ``aclose``."""

    def __init__(self, chunks=None):
        super().__init__(chunks=chunks)
        self.close_count = 0

    def close(self):
        self.close_count += 1


def test_async_stream_wrapper_prefers_aclose_over_sync_close():
    async def exercise():
        stream = _FakeBothCloseStream(chunks=["a"])
        wrapper = _TestAsyncStreamWrapper(stream)

        await wrapper.close()

        assert stream.aclose_count == 1
        assert stream.close_count == 0
        assert wrapper._self_stop_count == 1

    asyncio.run(exercise())


class _FakeClosable:
    def __init__(self, close_error=None):
        self.close_count = 0
        self.attribute = "passthrough"
        self._close_error = close_error

    def close(self):
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error

    async def aclose(self):
        self.close_count += 1
        if self._close_error is not None:
            raise self._close_error


def test_finalize_on_close_finalizes_and_forwards():
    closable = _FakeClosable()
    finalized = []

    proxy = finalize_on_close(closable, lambda: finalized.append(True))

    assert proxy.attribute == "passthrough"
    proxy.close()

    assert closable.close_count == 1
    assert finalized == [True]


def test_finalize_on_close_finalizes_when_close_raises():
    closable = _FakeClosable(close_error=RuntimeError("close failure"))
    finalized = []

    proxy = finalize_on_close(closable, lambda: finalized.append(True))

    with pytest.raises(RuntimeError, match="close failure"):
        proxy.close()

    assert finalized == [True]


def test_finalize_on_aclose_finalizes_and_forwards():
    async def exercise():
        closable = _FakeClosable()
        finalized = []

        proxy = finalize_on_aclose(closable, lambda: finalized.append(True))

        assert proxy.attribute == "passthrough"
        await proxy.aclose()

        assert closable.close_count == 1
        assert finalized == [True]

    asyncio.run(exercise())


def test_finalize_on_aclose_finalizes_when_aclose_raises():
    async def exercise():
        closable = _FakeClosable(close_error=RuntimeError("close failure"))
        finalized = []

        proxy = finalize_on_aclose(closable, lambda: finalized.append(True))

        with pytest.raises(RuntimeError, match="close failure"):
            await proxy.aclose()

        assert finalized == [True]

    asyncio.run(exercise())


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


class _FakeSyncManager:
    def __init__(
        self, stream, *, suppressed=False, enter_error=None, exit_error=None
    ):
        self._stream = stream
        self._suppressed = suppressed
        self._enter_error = enter_error
        self._exit_error = exit_error
        self.exit_args = None
        self.attribute = "passthrough"

    def __enter__(self):
        if self._enter_error is not None:
            raise self._enter_error
        return self._stream

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        if self._exit_error is not None:
            raise self._exit_error
        return self._suppressed


class _FakeAsyncManager(_FakeSyncManager):
    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return _FakeSyncManager.__exit__(self, exc_type, exc_val, exc_tb)


class _TestSyncManagerWrapper(SyncStreamManagerWrapper):
    def _wrap_stream(self, stream, invocation):
        return _TestSyncStreamWrapper(stream, invocation=invocation)


class _TestAsyncManagerWrapper(AsyncStreamManagerWrapper):
    def _wrap_stream(self, stream, invocation):
        return _TestAsyncStreamWrapper(stream, invocation=invocation)


class _ScopedEnterSyncManagerWrapper(_TestSyncManagerWrapper):
    """Runs the SDK entry inside instrumentation-owned state."""

    def _enter_manager(self, invocation):
        self._self_scope.append("enter")
        try:
            return super()._enter_manager(invocation)
        finally:
            self._self_scope.append("exit")


class _ScopedEnterAsyncManagerWrapper(_TestAsyncManagerWrapper):
    async def _enter_manager(self, invocation):
        self._self_scope.append("enter")
        try:
            return await super()._enter_manager(invocation)
        finally:
            self._self_scope.append("exit")


def test_sync_manager_wrapper_enters_and_finalizes_stream():
    stream = _FakeSyncStream(chunks=["a"])
    manager = _FakeSyncManager(stream)
    wrapper = _TestSyncManagerWrapper(manager, _FakeInvocation)

    with wrapper as stream_wrapper:
        assert stream_wrapper.__wrapped__ is stream

    assert manager.exit_args == (None, None, None)
    assert stream_wrapper._self_stop_count == 1


def test_sync_manager_wrapper_defers_invocation_until_enter():
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return _FakeInvocation()

    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream()), factory
    )

    assert not factory_calls

    with wrapper:
        pass

    assert factory_calls == [True]


def test_sync_manager_wrapper_forwards_attributes():
    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream()), _FakeInvocation
    )

    assert wrapper.attribute == "passthrough"


def test_sync_manager_wrapper_fails_invocation_when_enter_raises():
    error = RuntimeError("manager enter failure")
    invocation = _FakeInvocation()
    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream(), enter_error=error),
        lambda: invocation,
    )

    with pytest.raises(RuntimeError, match="manager enter failure"):
        with wrapper:
            pass

    assert invocation.failures == [error]


def test_sync_manager_wrapper_forwards_caller_error_to_stream_wrapper():
    manager = _FakeSyncManager(_FakeSyncStream())
    wrapper = _TestSyncManagerWrapper(manager, _FakeInvocation)
    error = ValueError("caller failure")

    with pytest.raises(ValueError, match="caller failure"):
        with wrapper as stream_wrapper:
            raise error

    assert manager.exit_args[:2] == (ValueError, error)
    assert stream_wrapper._self_failures == [error]


def test_sync_manager_wrapper_stops_stream_when_manager_suppresses():
    manager = _FakeSyncManager(_FakeSyncStream(), suppressed=True)
    wrapper = _TestSyncManagerWrapper(manager, _FakeInvocation)

    with wrapper as stream_wrapper:
        raise ValueError("suppressed")

    assert stream_wrapper._self_stop_count == 1
    assert not stream_wrapper._self_failures


def test_sync_manager_wrapper_finalizes_stream_when_exit_raises():
    manager_error = RuntimeError("manager failure")
    manager = _FakeSyncManager(_FakeSyncStream(), exit_error=manager_error)
    wrapper = _TestSyncManagerWrapper(manager, _FakeInvocation)

    stream_wrapper = wrapper.__enter__()
    with pytest.raises(RuntimeError, match="manager failure"):
        wrapper.__exit__(None, None, None)

    assert stream_wrapper._self_failures == [manager_error]


def test_sync_manager_wrapper_fails_invocation_when_exit_raises_before_enter():
    manager_error = RuntimeError("manager failure")
    invocation = _FakeInvocation()
    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream(), exit_error=manager_error),
        lambda: invocation,
    )
    wrapper._self_invocation = invocation

    with pytest.raises(RuntimeError, match="manager failure"):
        wrapper.__exit__(None, None, None)

    assert invocation.failures == [manager_error]


def test_async_manager_wrapper_enters_and_finalizes_stream():
    async def exercise():
        stream = _FakeAsyncStream(chunks=["a"])
        manager = _FakeAsyncManager(stream)
        wrapper = _TestAsyncManagerWrapper(manager, _FakeInvocation)

        async with wrapper as stream_wrapper:
            assert stream_wrapper.__wrapped__ is stream

        assert manager.exit_args == (None, None, None)
        assert stream_wrapper._self_stop_count == 1

    asyncio.run(exercise())


def test_async_manager_wrapper_fails_invocation_when_enter_raises():
    async def exercise():
        error = RuntimeError("manager enter failure")
        invocation = _FakeInvocation()
        wrapper = _TestAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream(), enter_error=error),
            lambda: invocation,
        )

        with pytest.raises(RuntimeError, match="manager enter failure"):
            async with wrapper:
                pass

        assert invocation.failures == [error]

    asyncio.run(exercise())


def test_async_manager_wrapper_forwards_caller_error_to_stream_wrapper():
    async def exercise():
        manager = _FakeAsyncManager(_FakeAsyncStream())
        wrapper = _TestAsyncManagerWrapper(manager, _FakeInvocation)
        error = ValueError("caller failure")

        with pytest.raises(ValueError, match="caller failure"):
            async with wrapper as stream_wrapper:
                raise error

        assert manager.exit_args[:2] == (ValueError, error)
        assert stream_wrapper._self_failures == [error]

    asyncio.run(exercise())


def test_async_manager_wrapper_stops_stream_when_manager_suppresses():
    async def exercise():
        manager = _FakeAsyncManager(_FakeAsyncStream(), suppressed=True)
        wrapper = _TestAsyncManagerWrapper(manager, _FakeInvocation)

        async with wrapper as stream_wrapper:
            raise ValueError("suppressed")

        assert stream_wrapper._self_stop_count == 1
        assert not stream_wrapper._self_failures

    asyncio.run(exercise())


def test_async_manager_wrapper_finalizes_stream_when_exit_raises():
    async def exercise():
        manager_error = RuntimeError("manager failure")
        manager = _FakeAsyncManager(
            _FakeAsyncStream(), exit_error=manager_error
        )
        wrapper = _TestAsyncManagerWrapper(manager, _FakeInvocation)

        stream_wrapper = await wrapper.__aenter__()
        with pytest.raises(RuntimeError, match="manager failure"):
            await wrapper.__aexit__(None, None, None)

        assert stream_wrapper._self_failures == [manager_error]

    asyncio.run(exercise())


def test_sync_manager_wrapper_hands_factory_invocation_to_stream_wrapper():
    invocation = _FakeInvocation()
    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream(chunks=["a"])), lambda: invocation
    )

    with wrapper as stream_wrapper:
        assert stream_wrapper._self_invocation is invocation
        assert invocation._request_stream is True
        assert invocation.stop_count == 0

    assert invocation.stop_count == 1
    assert not invocation.failures


def test_sync_manager_wrapper_fails_factory_invocation_on_caller_error():
    invocation = _FakeInvocation()
    wrapper = _TestSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream()), lambda: invocation
    )
    error = ValueError("caller failure")

    with pytest.raises(ValueError, match="caller failure"):
        with wrapper:
            raise error

    assert invocation.failures == [error]
    assert invocation.stop_count == 0


def test_sync_manager_wrapper_scopes_custom_enter_manager():
    wrapper = _ScopedEnterSyncManagerWrapper(
        _FakeSyncManager(_FakeSyncStream()), _FakeInvocation
    )
    wrapper._self_scope = []

    with wrapper:
        pass

    assert wrapper._self_scope == ["enter", "exit"]


def test_sync_manager_wrapper_unscopes_custom_enter_manager_on_failure():
    wrapper = _ScopedEnterSyncManagerWrapper(
        _FakeSyncManager(
            _FakeSyncStream(), enter_error=RuntimeError("enter failure")
        ),
        _FakeInvocation,
    )
    wrapper._self_scope = []

    with pytest.raises(RuntimeError, match="enter failure"):
        wrapper.__enter__()

    assert wrapper._self_scope == ["enter", "exit"]


def test_async_manager_wrapper_hands_factory_invocation_to_stream_wrapper():
    async def exercise():
        invocation = _FakeInvocation()
        wrapper = _TestAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream(chunks=["a"])),
            lambda: invocation,
        )

        async with wrapper as stream_wrapper:
            assert stream_wrapper._self_invocation is invocation
            assert invocation.stop_count == 0

        assert invocation.stop_count == 1
        assert not invocation.failures

    asyncio.run(exercise())


def test_async_manager_wrapper_scopes_custom_enter_manager():
    async def exercise():
        wrapper = _ScopedEnterAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream()), _FakeInvocation
        )
        wrapper._self_scope = []

        async with wrapper:
            pass

        assert wrapper._self_scope == ["enter", "exit"]

    asyncio.run(exercise())


def test_async_manager_wrapper_unscopes_custom_enter_manager_on_failure():
    async def exercise():
        wrapper = _ScopedEnterAsyncManagerWrapper(
            _FakeAsyncManager(
                _FakeAsyncStream(), enter_error=RuntimeError("enter failure")
            ),
            _FakeInvocation,
        )
        wrapper._self_scope = []

        with pytest.raises(RuntimeError, match="enter failure"):
            await wrapper.__aenter__()

        assert wrapper._self_scope == ["enter", "exit"]

    asyncio.run(exercise())


def test_async_manager_wrapper_defers_invocation_until_enter():
    async def exercise():
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return _FakeInvocation()

        wrapper = _TestAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream()), factory
        )

        assert not factory_calls

        async with wrapper:
            pass

        assert factory_calls == [True]

    asyncio.run(exercise())


def test_async_manager_wrapper_forwards_attributes():
    wrapper = _TestAsyncManagerWrapper(
        _FakeAsyncManager(_FakeAsyncStream()), _FakeInvocation
    )

    assert wrapper.attribute == "passthrough"


def test_async_manager_wrapper_fails_factory_invocation_on_caller_error():
    async def exercise():
        invocation = _FakeInvocation()
        wrapper = _TestAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream()), lambda: invocation
        )
        error = ValueError("caller failure")

        with pytest.raises(ValueError, match="caller failure"):
            async with wrapper:
                raise error

        assert invocation.failures == [error]
        assert invocation.stop_count == 0

    asyncio.run(exercise())


def test_async_manager_wrapper_fails_invocation_when_exit_raises_before_enter():
    async def exercise():
        manager_error = RuntimeError("manager failure")
        invocation = _FakeInvocation()
        wrapper = _TestAsyncManagerWrapper(
            _FakeAsyncManager(_FakeAsyncStream(), exit_error=manager_error),
            lambda: invocation,
        )
        wrapper._self_invocation = invocation

        with pytest.raises(RuntimeError, match="manager failure"):
            await wrapper.__aexit__(None, None, None)

        assert invocation.failures == [manager_error]

    asyncio.run(exercise())
