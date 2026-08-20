# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import timeit
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncIterable, Callable, Iterable
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from opentelemetry.util.genai._invocation import GenAIInvocation

    class _ObjectProxy:
        __wrapped__: Any

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy


ChunkT = TypeVar("ChunkT")
WrappedT = TypeVar("WrappedT")
SyncStreamWrapperT = TypeVar(
    "SyncStreamWrapperT", bound="SyncStreamWrapper[Any]"
)
AsyncStreamWrapperT = TypeVar(
    "AsyncStreamWrapperT", bound="AsyncStreamWrapper[Any]"
)
StreamT = TypeVar("StreamT")
InvocationT = TypeVar("InvocationT", bound="GenAIInvocation")
_ChunkT_co = TypeVar("_ChunkT_co", covariant=True)
_logger = logging.getLogger(__name__)


class _StreamWrapperMeta(ABCMeta, type(_ObjectProxy)):
    """Metaclass compatible with wrapt's proxy type and ABC hooks."""


class _SyncStream(Iterable[_ChunkT_co], Protocol[_ChunkT_co]):
    """Structural type for streams accepted by ``SyncStreamWrapper``."""

    def close(self) -> None: ...


class _AsyncStream(AsyncIterable[_ChunkT_co], Protocol[_ChunkT_co]):
    """Structural type for streams accepted by ``AsyncStreamWrapper``.

    No close method is required: SDK streams spell it ``close``, async
    generators spell it ``aclose``, and either is resolved at runtime by
    ``AsyncStreamWrapper._close_stream``.
    """


class _StreamTelemetry(Generic[ChunkT], metaclass=ABCMeta):
    """Finalize-once bookkeeping and telemetry hooks shared by both wrappers."""

    _self_finalized: bool

    def _finalize_success(self) -> None:
        if self._self_finalized:
            return
        self._self_finalized = True
        self._on_stream_end()

    def _finalize_failure(self, error: BaseException) -> None:
        if self._self_finalized:
            return
        self._self_finalized = True
        self._on_stream_error(error)

    @abstractmethod
    def _process_chunk(self, chunk: ChunkT) -> None:
        """Process one stream chunk for telemetry."""

    @abstractmethod
    def _on_stream_end(self) -> None:
        """Finalize the stream successfully."""

    @abstractmethod
    def _on_stream_error(self, error: BaseException) -> None:
        """Finalize the stream with failure."""


class SyncStreamWrapper(
    _StreamTelemetry[ChunkT],
    _ObjectProxy,
    Generic[ChunkT],
    metaclass=_StreamWrapperMeta,
):
    """Base class for synchronous instrumented stream wrappers.

    Subclass this when wrapping a provider SDK stream that is consumed with
    normal iteration. The subclass should pass the SDK stream to
    ``super().__init__(stream)`` and implement the three telemetry hooks:
    ``_process_chunk`` for per-chunk state, ``_on_stream_end`` for successful
    finalization, and ``_on_stream_error`` for failure finalization.

    Users should consume subclasses as normal streams, for example with
    ``for chunk in wrapper`` or ``with wrapper``. The hook methods are called
    internally by the wrapper lifecycle and are not part of the public API.
    """

    def __init__(
        self,
        stream: _SyncStream[ChunkT],
        invocation: GenAIInvocation | None = None,
    ):
        super().__init__(stream)
        self._self_finalized = False
        # Marks the request as streamed (gen_ai.request.stream) and receives
        # per-chunk timing via _on_stream_chunk.
        self._self_invocation = invocation
        if invocation is not None:
            invocation._request_stream = True
        self._bind_stream(stream)

    # The SDK stream, held loosely typed: subclasses re-expose it through a
    # ``stream`` property carrying the SDK's own type.
    _self_stream: Any

    def _bind_stream(self, stream: _SyncStream[ChunkT]) -> None:
        self._self_stream = stream
        self._self_iterator = iter(stream)

    def _set_stream(self, stream: _SyncStream[ChunkT]) -> None:
        """Point the wrapper at ``stream``, rebinding proxy and iterator.

        Subclasses exposing a settable ``stream`` property (SDKs that swap the
        underlying stream after construction) delegate to this so the proxy
        target and the iterator driving ``__next__`` stay in sync.
        """
        self.__wrapped__ = stream
        self._bind_stream(stream)

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is not None:
            self._finalize_failure(exc_val)
            try:
                self._self_stream.close()
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "GenAI stream close error after user exception",
                    exc_info=True,
                )
            return False

        self.close()
        return False

    def close(self) -> None:
        try:
            self._self_stream.close()
        except Exception as error:
            self._finalize_failure(error)
            raise
        self._finalize_success()

    def __iter__(self):
        # Override ``ObjectProxy.__iter__`` so iteration drives ``__next__``
        # below and runs ``_process_chunk`` per chunk; otherwise iteration
        # would be forwarded to the wrapped stream and bypass instrumentation.
        return self

    def __next__(self) -> ChunkT:
        try:
            chunk = next(self._self_iterator)
        except StopIteration:
            self._finalize_success()
            raise
        except Exception as error:
            self._finalize_failure(error)
            raise
        invocation = self._self_invocation
        chunk_at = timeit.default_timer() if invocation is not None else None
        self._process_chunk(chunk)
        # Record after _process_chunk so response.model is on the metrics.
        if invocation is not None and chunk_at is not None:
            invocation._on_stream_chunk(chunk_at)
        return chunk


class AsyncStreamWrapper(
    _StreamTelemetry[ChunkT],
    _ObjectProxy,
    Generic[ChunkT],
    metaclass=_StreamWrapperMeta,
):
    """Base class for asynchronous instrumented stream wrappers.

    Subclass this when wrapping a provider SDK stream that is consumed with
    async iteration. The subclass should pass the SDK stream to
    ``super().__init__(stream)`` and implement the three telemetry hooks:
    ``_process_chunk`` for per-chunk state, ``_on_stream_end`` for successful
    finalization, and ``_on_stream_error`` for failure finalization.

    Users should consume subclasses as normal async streams, for example with
    ``async for chunk in wrapper`` or ``async with wrapper``. The hook methods
    remain synchronous telemetry hooks; async stream reads and close handling
    are owned by this base class.
    """

    def __init__(
        self,
        stream: _AsyncStream[ChunkT],
        invocation: GenAIInvocation | None = None,
    ):
        super().__init__(stream)
        self._self_finalized = False
        # Marks the request as streamed (gen_ai.request.stream) and receives
        # per-chunk timing via _on_stream_chunk.
        self._self_invocation = invocation
        if invocation is not None:
            invocation._request_stream = True
        self._bind_stream(stream)

    # See ``SyncStreamWrapper._self_stream``.
    _self_stream: Any

    def _bind_stream(self, stream: _AsyncStream[ChunkT]) -> None:
        self._self_stream = stream
        self._self_aiter = aiter(stream)

    def _set_stream(self, stream: _AsyncStream[ChunkT]) -> None:
        """Point the wrapper at ``stream``, rebinding proxy and iterator.

        See ``SyncStreamWrapper._set_stream``.
        """
        self.__wrapped__ = stream
        self._bind_stream(stream)

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is not None:
            self._finalize_failure(exc_val)
            try:
                await self._close_stream()
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "GenAI stream close error after user exception",
                    exc_info=True,
                )
            return False

        await self._close()
        return False

    async def _close_stream(self) -> None:
        """Close the wrapped stream, whichever close method it exposes.

        SDK streams (OpenAI's and Anthropic's ``AsyncStream``) expose an async
        ``close``; async generators -- what Google's async
        ``generate_content_stream`` hands back -- expose ``aclose`` instead.
        ``aclose`` is preferred for an object carrying both, because a pairing
        of the two names is how objects with a *sync* ``close`` spell their
        async one, and awaiting the sync one would raise.

        A stream exposing neither is left alone; there is nothing to close, and
        the caller's iteration has already ended.
        """
        close = getattr(self._self_stream, "aclose", None)
        if close is None:
            close = getattr(self._self_stream, "close", None)
        if close is None:
            return
        await close()

    async def _close(self) -> None:
        """Close the stream and finalize telemetry.

        Reached through whichever of ``close`` / ``aclose`` the wrapped stream
        exposes; see ``__getattr__``.
        """
        try:
            await self._close_stream()
        except Exception as error:
            self._finalize_failure(error)
            _logger.debug(
                "GenAI stream close error during close",
                exc_info=True,
            )
            raise
        self._finalize_success()

    if TYPE_CHECKING:
        # Declared for type checkers only. Defining them for real would make
        # the wrapper advertise both names at runtime, and a ``__getattr__``
        # visible to the checker would type every forwarded attribute as
        # ``Any`` -- silently disabling attribute checking on every subclass.
        async def close(self) -> None: ...

        async def aclose(self) -> None: ...

    else:

        def __getattr__(self, name):
            # Forwards to the wrapped stream like
            # ``wrapt.ObjectProxy.__getattr__``, except for the close method the
            # stream itself exposes: that one is served from here so the call
            # finalizes telemetry instead of reaching the stream directly.
            # Serving it here rather than defining ``close`` / ``aclose`` on the
            # class keeps the wrapper from advertising the name its stream
            # doesn't have -- async streams are routinely duck-typed on exactly
            # that shape (an ``AsyncStream`` has ``close``, an async generator
            # has ``aclose``).
            wrapped = object.__getattribute__(self, "__wrapped__")
            if name in ("close", "aclose") and hasattr(wrapped, name):
                return self._close
            return getattr(wrapped, name)

    def __aiter__(self):
        # Override ``ObjectProxy.__aiter__`` so iteration drives ``__anext__``
        # below and runs ``_process_chunk`` per chunk; otherwise iteration
        # would be forwarded to the wrapped stream and bypass instrumentation.
        return self

    async def __anext__(self) -> ChunkT:
        try:
            chunk = await anext(self._self_aiter)
        except StopAsyncIteration:
            self._finalize_success()
            raise
        except Exception as error:
            self._finalize_failure(error)
            raise

        invocation = self._self_invocation
        chunk_at = timeit.default_timer() if invocation is not None else None
        self._process_chunk(chunk)
        # Record after _process_chunk so response.model is on the metrics.
        if invocation is not None and chunk_at is not None:
            invocation._on_stream_chunk(chunk_at)
        return chunk


class _CloseFinalizingProxy(_ObjectProxy):
    def __init__(self, wrapped: object, finalize: Callable[[], None]) -> None:
        super().__init__(wrapped)
        self._self_finalize = finalize

    def close(self) -> None:
        try:
            self.__wrapped__.close()
        finally:
            self._self_finalize()


class _AcloseFinalizingProxy(_ObjectProxy):
    def __init__(self, wrapped: object, finalize: Callable[[], None]) -> None:
        super().__init__(wrapped)
        self._self_finalize = finalize

    async def aclose(self) -> None:
        try:
            await self.__wrapped__.aclose()
        finally:
            self._self_finalize()


def finalize_on_close(
    wrapped: WrappedT, finalize: Callable[[], None]
) -> WrappedT:
    """Proxy ``wrapped`` so closing it also finalizes telemetry.

    Used for objects an SDK stream exposes to callers as a way to end the
    operation early -- typically the underlying HTTP response behind
    ``stream.response`` -- where a ``close()`` means the caller is done and the
    invocation should be finalized. Everything but ``close`` forwards
    unchanged.
    """
    return cast(WrappedT, _CloseFinalizingProxy(wrapped, finalize))


def finalize_on_aclose(
    wrapped: WrappedT, finalize: Callable[[], None]
) -> WrappedT:
    """Async counterpart of ``finalize_on_close``, hooking ``aclose``."""
    return cast(WrappedT, _AcloseFinalizingProxy(wrapped, finalize))


class SyncStreamManagerWrapper(
    _ObjectProxy,
    Generic[StreamT, InvocationT, SyncStreamWrapperT],
    metaclass=_StreamWrapperMeta,
):
    """Base class for wrappers of SDK stream *managers*.

    A stream manager is the context manager an SDK returns from a ``stream()``
    style API: entering it opens the HTTP request and yields the stream, and
    exiting it closes the stream. The invocation is created on entry -- not
    when the manager is constructed -- so a manager that is never entered opens
    no span.

    Subclasses implement ``_wrap_stream`` to return the instrumented stream
    wrapper for the SDK stream, and may override ``_enter_manager`` when the
    SDK's entry needs to run inside instrumentation-specific state. The type
    parameters are the SDK's stream type, the invocation type the factory
    produces, and the stream wrapper type, so neither hook needs a cast.
    """

    def __init__(
        self,
        manager: object,
        invocation_factory: Callable[[], InvocationT],
    ) -> None:
        super().__init__(manager)
        self._self_invocation_factory = invocation_factory
        self._self_invocation: InvocationT | None = None
        self._self_stream_wrapper: SyncStreamWrapperT | None = None

    def _enter_manager(self, invocation: InvocationT) -> StreamT:
        """Enter the wrapped SDK manager and return its stream."""
        return cast(StreamT, self.__wrapped__.__enter__())

    @abstractmethod
    def _wrap_stream(
        self, stream: StreamT, invocation: InvocationT
    ) -> SyncStreamWrapperT:
        """Return the instrumented wrapper for the SDK's stream."""

    def __enter__(self) -> SyncStreamWrapperT:
        invocation = self._self_invocation_factory()
        self._self_invocation = invocation
        try:
            stream = self._enter_manager(invocation)
        except Exception as error:
            invocation.fail(error)
            raise
        stream_wrapper = self._wrap_stream(stream, invocation)
        self._self_stream_wrapper = stream_wrapper
        return stream_wrapper

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        stream_wrapper = self._self_stream_wrapper
        self._self_stream_wrapper = None
        try:
            suppressed = self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        except Exception as error:
            if stream_wrapper is not None:
                stream_wrapper.__exit__(
                    type(error), error, error.__traceback__
                )
            elif self._self_invocation is not None:
                self._self_invocation.fail(error)
            raise
        if stream_wrapper is not None:
            if suppressed:
                # The manager swallowed the caller's exception, so the stream
                # ended successfully as far as telemetry is concerned.
                stream_wrapper.__exit__(None, None, None)
            else:
                stream_wrapper.__exit__(exc_type, exc_val, exc_tb)
        return suppressed


class AsyncStreamManagerWrapper(
    _ObjectProxy,
    Generic[StreamT, InvocationT, AsyncStreamWrapperT],
    metaclass=_StreamWrapperMeta,
):
    """Async counterpart of ``SyncStreamManagerWrapper``."""

    def __init__(
        self,
        manager: object,
        invocation_factory: Callable[[], InvocationT],
    ) -> None:
        super().__init__(manager)
        self._self_invocation_factory = invocation_factory
        self._self_invocation: InvocationT | None = None
        self._self_stream_wrapper: AsyncStreamWrapperT | None = None

    async def _enter_manager(self, invocation: InvocationT) -> StreamT:
        """Enter the wrapped SDK manager and return its stream."""
        return cast(StreamT, await self.__wrapped__.__aenter__())

    @abstractmethod
    def _wrap_stream(
        self, stream: StreamT, invocation: InvocationT
    ) -> AsyncStreamWrapperT:
        """Return the instrumented wrapper for the SDK's stream."""

    async def __aenter__(self) -> AsyncStreamWrapperT:
        invocation = self._self_invocation_factory()
        self._self_invocation = invocation
        try:
            stream = await self._enter_manager(invocation)
        except Exception as error:
            invocation.fail(error)
            raise
        stream_wrapper = self._wrap_stream(stream, invocation)
        self._self_stream_wrapper = stream_wrapper
        return stream_wrapper

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        stream_wrapper = self._self_stream_wrapper
        self._self_stream_wrapper = None
        try:
            suppressed = await self.__wrapped__.__aexit__(
                exc_type, exc_val, exc_tb
            )
        except Exception as error:
            if stream_wrapper is not None:
                await stream_wrapper.__aexit__(
                    type(error), error, error.__traceback__
                )
            elif self._self_invocation is not None:
                self._self_invocation.fail(error)
            raise
        if stream_wrapper is not None:
            if suppressed:
                # See SyncStreamManagerWrapper.__exit__.
                await stream_wrapper.__aexit__(None, None, None)
            else:
                await stream_wrapper.__aexit__(exc_type, exc_val, exc_tb)
        return suppressed


__all__ = [
    "AsyncStreamManagerWrapper",
    "AsyncStreamWrapper",
    "SyncStreamManagerWrapper",
    "SyncStreamWrapper",
    "finalize_on_aclose",
    "finalize_on_close",
]
