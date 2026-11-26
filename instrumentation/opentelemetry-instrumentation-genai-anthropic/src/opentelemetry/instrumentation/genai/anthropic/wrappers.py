# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
)
from contextlib import AbstractContextManager, contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
)

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib

from opentelemetry.util.genai.stream import (
    AsyncStreamManagerWrapper,
    AsyncStreamWrapper,
    SyncStreamManagerWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
)

from .messages_extractors import (
    get_input_messages,
    get_system_instruction,
    get_tool_definitions,
    set_invocation_response_attributes,
)

_logger = logging.getLogger(__name__)

try:
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        accumulate_event as _sdk_accumulate_event,
    )
except ImportError:
    _sdk_accumulate_event = None

try:
    from anthropic.lib.streaming._beta_messages import (  # pylint: disable=no-name-in-module
        accumulate_event as _sdk_beta_accumulate_event,
    )
except ImportError:
    _sdk_beta_accumulate_event = None

if TYPE_CHECKING:
    from anthropic._streaming import AsyncStream, Stream
    from anthropic.lib.streaming._beta_messages import (  # pylint: disable=no-name-in-module
        BetaAsyncMessageStream,
        BetaAsyncMessageStreamManager,
        BetaMessageStream,
        BetaMessageStreamManager,
    )
    from anthropic.lib.streaming._beta_types import (  # pylint: disable=no-name-in-module
        ParsedBetaMessageStreamEvent,
    )
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        AsyncMessageStream,
        AsyncMessageStreamManager,
        MessageStream,
        MessageStreamManager,
    )
    from anthropic.lib.streaming._types import (  # pylint: disable=no-name-in-module
        ParsedMessageStreamEvent,
    )
    from anthropic.types import (
        Message,
        RawMessageStreamEvent,
    )
    from anthropic.types.beta import (
        BetaMessage,
        BetaRawMessageStreamEvent,
    )
    from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
    from anthropic.types.parsed_message import ParsedMessage

    from opentelemetry.util.genai.invocation import InferenceInvocation
    from opentelemetry.util.genai.iterator_capture import IteratorCapture

    # wrapt's ``ObjectProxy`` is untyped; model just the constructor and the
    # wrapped-object attribute we use so the proxy subclass is fully typed
    # under pyright.
    class _ObjectProxy:
        __wrapped__: Any

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy


ResponseFormatT = TypeVar("ResponseFormatT")
_SourceT = TypeVar("_SourceT")
_ViewT = TypeVar("_ViewT")
accumulate_event = cast("Callable[..., Message] | None", _sdk_accumulate_event)
beta_accumulate_event = cast(
    "Callable[..., ParsedBetaMessage[Any]] | None", _sdk_beta_accumulate_event
)

_accumulate_takes_json_bufs = False
if accumulate_event is not None:
    try:
        _accumulate_takes_json_bufs = (
            "json_bufs" in inspect.signature(accumulate_event).parameters
        )
    except (ValueError, TypeError):
        _accumulate_takes_json_bufs = False

_beta_accumulate_accepts_headers = False
_beta_accumulate_takes_json_bufs = False
if beta_accumulate_event is not None:
    try:
        _beta_accumulate_parameters = inspect.signature(
            beta_accumulate_event
        ).parameters
        _beta_accumulate_accepts_headers = (
            "request_headers" in _beta_accumulate_parameters
        )
        _beta_accumulate_takes_json_bufs = (
            "json_bufs" in _beta_accumulate_parameters
        )
    except (ValueError, TypeError):
        pass

_accumulation_disabled = False

# ``MessageStream`` helpers that drain the SDK stream through the SDK object
# itself, so they never pass through the wrapper's scoped reads.
_DRAINING_HELPERS = frozenset(
    ("until_done", "get_final_message", "get_final_text")
)


def _close_text_source(text_stream: Iterator[str]) -> None:
    """Close the SDK generator behind a scoped ``text_stream``."""
    if isinstance(text_stream, Generator):
        text_stream.close()


async def _aclose_text_source(text_stream: AsyncIterator[str]) -> None:
    """Async counterpart of ``_close_text_source``."""
    if isinstance(text_stream, AsyncGenerator):
        await text_stream.aclose()


class _TextStreamScope(Protocol):
    def _reading(self) -> AbstractContextManager[None]: ...


class _ScopedTextStream(_ObjectProxy):
    """The SDK's ``text_stream`` generator, read and closed under the invocation.

    A generator adapter can't stand in for it: closing one before its first
    read never runs its body, so the SDK generator would stay open.
    """

    __wrapped__: Iterator[str]

    def __init__(
        self, wrapped: Iterator[str], scope: _TextStreamScope
    ) -> None:
        super().__init__(wrapped)
        self._self_scope = scope

    def __iter__(self) -> Iterator[str]:
        return self

    def _read(self, read: Callable[[], str]) -> str:
        with self._self_scope._reading():
            try:
                return read()
            except StopIteration:
                pass
        raise StopIteration

    def __next__(self) -> str:
        return self._read(lambda: next(self.__wrapped__))

    def send(self, value: Any) -> str:
        source = cast("Generator[str, Any, Any]", self.__wrapped__)
        return self._read(lambda: source.send(value))

    def throw(self, *args: Any) -> str:
        source = cast("Generator[str, Any, Any]", self.__wrapped__)
        return self._read(lambda: source.throw(*args))

    def close(self) -> None:
        with self._self_scope._reading():
            _close_text_source(self.__wrapped__)


class _AsyncScopedTextStream(_ObjectProxy):
    """Async counterpart of ``_ScopedTextStream``."""

    __wrapped__: AsyncIterator[str]

    def __init__(
        self, wrapped: AsyncIterator[str], scope: _TextStreamScope
    ) -> None:
        super().__init__(wrapped)
        self._self_scope = scope

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def _read(self, read: Callable[[], Awaitable[str]]) -> str:
        with self._self_scope._reading():
            try:
                return await read()
            except StopAsyncIteration:
                pass
        raise StopAsyncIteration

    async def __anext__(self) -> str:
        return await self._read(lambda: anext(self.__wrapped__))

    async def asend(self, value: Any) -> str:
        source = cast("AsyncGenerator[str, Any]", self.__wrapped__)
        return await self._read(lambda: source.asend(value))

    async def athrow(self, *args: Any) -> str:
        source = cast("AsyncGenerator[str, Any]", self.__wrapped__)
        return await self._read(lambda: source.athrow(*args))

    async def aclose(self) -> None:
        with self._self_scope._reading():
            await _aclose_text_source(self.__wrapped__)


class _ReadScope(Protocol):
    def _execution_context(self) -> AbstractContextManager[None]: ...

    def _finalize_failure(self, error: BaseException) -> None: ...


class _StreamWrapperWithStream(Protocol):
    @property
    def stream(self) -> object: ...


def _set_response_attributes(
    invocation: InferenceInvocation,
    result: (
        Message
        | BetaMessage
        | ParsedMessage[Any]
        | ParsedBetaMessage[Any]
        | None
    ),
    capture_content: bool,
) -> None:
    set_invocation_response_attributes(invocation, result, capture_content)


class MessageWrapper:
    """Wrapper for non-streaming Message response that handles telemetry."""

    def __init__(self, message: Message | BetaMessage, capture_content: bool):
        self._message = message
        self._capture_content = capture_content

    def extract_into(self, invocation: InferenceInvocation) -> None:
        """Extract response data into the invocation."""
        set_invocation_response_attributes(
            invocation, self._message, self._capture_content
        )

    @property
    def message(self) -> Message | BetaMessage:
        """Return the wrapped Message object."""
        return self._message


class _MessagesStreamMixin(Generic[ResponseFormatT]):
    _self_invocation: InferenceInvocation
    _self_message: (
        Message
        | BetaMessage
        | ParsedMessage[ResponseFormatT]
        | ParsedBetaMessage[ResponseFormatT]
        | None
    )
    _self_capture_content: bool
    _self_message_telemetry_finalized: bool
    _self_json_bufs: dict[int, bytes]
    _self_is_beta: bool
    _self_beta_accumulation_disabled: bool
    # The SDK hands out the same text generator every time, so the scoped
    # copy of it is memoized against the generator it was built from.
    _self_text_stream_source: object
    _self_text_stream: object

    if not TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any:
            attribute = super().__getattr__(name)
            if name == "text_stream":
                return self._scoped_text_stream(attribute)
            if name in _DRAINING_HELPERS:
                return self._scoped_helper(name, attribute)
            return attribute

    def _memoized_text_view(
        self, text_stream: _SourceT, view: Callable[[_SourceT], _ViewT]
    ) -> _ViewT:
        if self._self_text_stream_source is not text_stream:
            self._self_text_stream_source = text_stream
            self._self_text_stream = view(text_stream)
        return cast(_ViewT, self._self_text_stream)

    def _drain_for(self, name: str) -> Callable[[], object] | None:
        # Only draining reads the response; the final-message helpers then
        # raise their own usage errors on a successful response
        # (``get_final_text()`` on a reply with no text block), which must not
        # fail the span. So drain under the invocation, then call the helper.
        if name == "until_done":
            return None
        stream = cast(_StreamWrapperWithStream, self).stream
        drain: Callable[[], object] | None = getattr(
            stream, "until_done", None
        )
        return drain

    @contextmanager
    def _reading(self) -> Iterator[None]:
        # Finalize through the wrapper before ``activate()`` fails the
        # invocation, so the SDK snapshot is still recorded.
        scope = cast(_ReadScope, self)
        with scope._execution_context():
            try:
                yield
            except BaseException as error:
                scope._finalize_failure(error)
                raise

    def _stop(self) -> None:
        if self._self_message_telemetry_finalized:
            return
        # text_stream and the get_final_* helpers bypass _process_chunk, so the
        # snapshot can be the only record of the response.
        self._adopt_sdk_snapshot()
        _set_response_attributes(
            self._self_invocation,
            self._self_message,
            self._self_capture_content,
        )
        self._self_invocation.stop()
        self._self_message_telemetry_finalized = True

    def _fail(self, exc: BaseException) -> None:
        if self._self_message_telemetry_finalized:
            return
        self._adopt_sdk_snapshot()
        message = self._self_message
        if message is not None:
            capture_content = self._self_capture_content and bool(
                message.content
            )
            _set_response_attributes(
                self._self_invocation, message, capture_content
            )
        self._self_invocation.fail(exc)
        self._self_message_telemetry_finalized = True

    def _on_stream_end(self) -> None:
        self._stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._fail(error)

    def _adopt_sdk_snapshot(self) -> bool:
        """Adopt the SDK stream's accumulated message, when it keeps one.

        ``MessageStream`` updates ``current_message_snapshot`` as the response
        arrives, whichever accessor the caller reads it through. A plain
        ``Stream`` has no snapshot and leaves accumulation to us.
        """
        stream = cast(_StreamWrapperWithStream, self).stream
        try:
            snapshot = cast(
                "ParsedMessage[ResponseFormatT] | None",
                getattr(stream, "current_message_snapshot", None),
            )
        except AssertionError:
            # The property asserts the snapshot is set, so an unconsumed stream
            # raises rather than answering None.
            return False
        if snapshot is None:
            return False
        self._self_message = snapshot
        return True

    def _process_chunk(
        self,
        chunk: (
            RawMessageStreamEvent
            | BetaRawMessageStreamEvent
            | ParsedMessageStreamEvent[ResponseFormatT]
            | ParsedBetaMessageStreamEvent[ResponseFormatT]
        ),
    ) -> None:
        """Accumulate a final message snapshot from a streaming chunk."""
        global _accumulation_disabled
        if self._adopt_sdk_snapshot():
            return
        stream = cast(_StreamWrapperWithStream, self).stream
        is_beta = self._self_is_beta or getattr(
            chunk.__class__, "__module__", ""
        ).startswith("anthropic.types.beta")
        if (
            is_beta
            and beta_accumulate_event is not None
            and not self._self_beta_accumulation_disabled
        ):
            beta_kwargs: dict[str, Any] = {
                "event": chunk,
                "current_snapshot": self._self_message,
            }
            if _beta_accumulate_takes_json_bufs:
                beta_kwargs["json_bufs"] = self._self_json_bufs
            if _beta_accumulate_accepts_headers:
                response = getattr(stream, "response", None)
                request = getattr(response, "request", None)
                headers = getattr(request, "headers", None)
                beta_kwargs["request_headers"] = (
                    headers if headers is not None else _http_lib.Headers()
                )
            try:
                self._self_message = beta_accumulate_event(**beta_kwargs)
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                self._self_beta_accumulation_disabled = True
                _logger.debug(
                    "Failed to accumulate beta stream event", exc_info=True
                )
            else:
                return

        if accumulate_event is None or _accumulation_disabled:
            return

        kwargs: dict[str, Any] = {
            "event": cast("RawMessageStreamEvent", chunk),
            "current_snapshot": cast(
                "ParsedMessage[ResponseFormatT] | None", self._self_message
            ),
        }
        if _accumulate_takes_json_bufs:
            kwargs["json_bufs"] = self._self_json_bufs

        try:
            self._self_message = accumulate_event(**kwargs)
        except BaseException as exc:
            _accumulation_disabled = True
            if not isinstance(exc, Exception):
                raise
            _logger.warning(
                "Failed to accumulate streaming content; this Anthropic SDK "
                "version is not supported. Future content accumulation is "
                "suppressed; please upgrade opentelemetry-instrumentation-genai-anthropic "
                "or report an issue.",
                exc_info=True,
            )


class MessagesStreamWrapper(
    _MessagesStreamMixin[ResponseFormatT],
    SyncStreamWrapper[
        "RawMessageStreamEvent | BetaRawMessageStreamEvent | ParsedMessageStreamEvent[ResponseFormatT] | ParsedBetaMessageStreamEvent[ResponseFormatT]"
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for Anthropic Stream that handles telemetry."""

    def __init__(
        self,
        stream: (
            Stream[RawMessageStreamEvent]
            | Stream[BetaRawMessageStreamEvent]
            | MessageStream[ResponseFormatT]
            | BetaMessageStream[ResponseFormatT]
        ),
        invocation: InferenceInvocation,
        capture_content: bool,
        is_beta: bool = False,
    ):
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_message = None
        self._self_capture_content = capture_content
        self._self_message_telemetry_finalized = False
        self._self_json_bufs = {}
        self._self_is_beta = is_beta
        self._self_beta_accumulation_disabled = False
        self._self_text_stream_source = None
        self._self_text_stream = None

    def _scoped_text_stream(self, text_stream: Iterator[str]) -> Iterator[str]:
        return self._memoized_text_view(
            text_stream, lambda source: _ScopedTextStream(source, self)
        )

    def _scoped_helper(
        self, name: str, helper: Callable[[], object]
    ) -> Callable[[], object]:
        drain = self._drain_for(name)

        # The SDK helpers take no arguments, so an invalid call raises here,
        # before anything is drained.
        @functools.wraps(helper)
        def scoped() -> object:
            with self._reading():
                if drain is None:
                    return helper()
                drain()
            return helper()

        return scoped

    @property
    def response(self) -> _http_lib.Response:
        return finalize_on_close(
            self.stream.response, self._stop, self._execution_context
        )

    @property
    def stream(
        self,
    ) -> (
        Stream[RawMessageStreamEvent]
        | Stream[BetaRawMessageStreamEvent]
        | MessageStream[ResponseFormatT]
        | BetaMessageStream[ResponseFormatT]
    ):
        return self._self_stream

    @stream.setter
    def stream(
        self,
        stream: (
            Stream[RawMessageStreamEvent]
            | Stream[BetaRawMessageStreamEvent]
            | MessageStream[ResponseFormatT]
            | BetaMessageStream[ResponseFormatT]
        ),
    ) -> None:
        self._set_stream(stream)


class AsyncMessagesStreamWrapper(
    _MessagesStreamMixin[ResponseFormatT],
    AsyncStreamWrapper[
        "RawMessageStreamEvent | BetaRawMessageStreamEvent | ParsedMessageStreamEvent[ResponseFormatT] | ParsedBetaMessageStreamEvent[ResponseFormatT]"
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for async Anthropic Stream that handles telemetry."""

    def __init__(
        self,
        stream: (
            AsyncStream[RawMessageStreamEvent]
            | AsyncStream[BetaRawMessageStreamEvent]
            | AsyncMessageStream[ResponseFormatT]
            | BetaAsyncMessageStream[ResponseFormatT]
        ),
        invocation: InferenceInvocation,
        capture_content: bool,
        is_beta: bool = False,
    ):
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_message = None
        self._self_capture_content = capture_content
        self._self_message_telemetry_finalized = False
        self._self_json_bufs = {}
        self._self_is_beta = is_beta
        self._self_beta_accumulation_disabled = False
        self._self_text_stream_source = None
        self._self_text_stream = None

    def _scoped_text_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        return self._memoized_text_view(
            text_stream, lambda source: _AsyncScopedTextStream(source, self)
        )

    def _scoped_helper(
        self, name: str, helper: Callable[[], Awaitable[object]]
    ) -> Callable[[], Awaitable[object]]:
        drain = self._drain_for(name)

        # See ``MessagesStreamWrapper._scoped_helper``.
        @functools.wraps(helper)
        async def scoped() -> object:
            with self._reading():
                if drain is None:
                    return await helper()
                await cast("Awaitable[None]", drain())
            return await helper()

        return scoped

    @property
    def response(self) -> _http_lib.Response:
        return finalize_on_aclose(
            self.stream.response, self._stop, self._execution_context
        )

    @property
    def stream(
        self,
    ) -> (
        AsyncStream[RawMessageStreamEvent]
        | AsyncStream[BetaRawMessageStreamEvent]
        | AsyncMessageStream[ResponseFormatT]
        | BetaAsyncMessageStream[ResponseFormatT]
    ):
        return self._self_stream

    @stream.setter
    def stream(
        self,
        stream: (
            AsyncStream[RawMessageStreamEvent]
            | AsyncStream[BetaRawMessageStreamEvent]
            | AsyncMessageStream[ResponseFormatT]
            | BetaAsyncMessageStream[ResponseFormatT]
        ),
    ) -> None:
        self._set_stream(stream)


class MessagesStreamManagerWrapper(
    SyncStreamManagerWrapper[
        "MessageStream[ResponseFormatT] | BetaMessageStream[ResponseFormatT]",
        "InferenceInvocation",
        "MessagesStreamWrapper[ResponseFormatT]",
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for sync Anthropic stream managers."""

    def __init__(
        self,
        manager: (
            MessageStreamManager[ResponseFormatT]
            | BetaMessageStreamManager[ResponseFormatT]
        ),
        invocation_factory: Callable[[], InferenceInvocation],
        capture_content: bool,
        *,
        messages_capture: IteratorCapture,
        system_capture: IteratorCapture,
        tools_capture: IteratorCapture,
        is_beta: bool = False,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content
        self._self_is_beta = is_beta
        self._self_messages_capture = messages_capture
        self._self_system_capture = system_capture
        self._self_tools_capture = tools_capture

    def _enter_manager(
        self, invocation: InferenceInvocation
    ) -> MessageStream[ResponseFormatT] | BetaMessageStream[ResponseFormatT]:
        with invocation.activate():
            try:
                return super()._enter_manager(invocation)
            finally:
                invocation.input_messages = (
                    self._self_messages_capture.extract(get_input_messages)
                    or []
                )
                invocation.system_instruction = (
                    self._self_system_capture.extract(get_system_instruction)
                    or []
                )
                invocation.tool_definitions = self._self_tools_capture.extract(
                    get_tool_definitions
                )

    def _wrap_stream(
        self,
        stream: (
            MessageStream[ResponseFormatT] | BetaMessageStream[ResponseFormatT]
        ),
        invocation: InferenceInvocation,
    ) -> MessagesStreamWrapper[ResponseFormatT]:
        return MessagesStreamWrapper(
            stream,
            invocation,
            self._self_capture_content,
            is_beta=self._self_is_beta,
        )


class AsyncMessagesStreamManagerWrapper(
    AsyncStreamManagerWrapper[
        "AsyncMessageStream[ResponseFormatT] | BetaAsyncMessageStream[ResponseFormatT]",
        "InferenceInvocation",
        "AsyncMessagesStreamWrapper[ResponseFormatT]",
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for AsyncMessageStreamManager that handles telemetry.

    Wraps AsyncMessageStreamManager from the Anthropic SDK:
    https://github.com/anthropics/anthropic-sdk-python/blob/05220bc1c1079fe01f5c4babc007ec7a990859d9/src/anthropic/lib/streaming/_messages.py#L294
    """

    def __init__(
        self,
        manager: (
            AsyncMessageStreamManager[ResponseFormatT]
            | BetaAsyncMessageStreamManager[ResponseFormatT]
        ),
        invocation_factory: Callable[[], InferenceInvocation],
        capture_content: bool,
        *,
        messages_capture: IteratorCapture,
        system_capture: IteratorCapture,
        tools_capture: IteratorCapture,
        is_beta: bool = False,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content
        self._self_is_beta = is_beta
        self._self_messages_capture = messages_capture
        self._self_system_capture = system_capture
        self._self_tools_capture = tools_capture

    async def _enter_manager(
        self, invocation: InferenceInvocation
    ) -> (
        AsyncMessageStream[ResponseFormatT]
        | BetaAsyncMessageStream[ResponseFormatT]
    ):
        with invocation.activate():
            try:
                return await super()._enter_manager(invocation)
            finally:
                invocation.input_messages = (
                    self._self_messages_capture.extract(get_input_messages)
                    or []
                )
                invocation.system_instruction = (
                    self._self_system_capture.extract(get_system_instruction)
                    or []
                )
                invocation.tool_definitions = self._self_tools_capture.extract(
                    get_tool_definitions
                )

    def _wrap_stream(
        self,
        stream: (
            AsyncMessageStream[ResponseFormatT]
            | BetaAsyncMessageStream[ResponseFormatT]
        ),
        invocation: InferenceInvocation,
    ) -> AsyncMessagesStreamWrapper[ResponseFormatT]:
        return AsyncMessagesStreamWrapper(
            stream,
            invocation,
            self._self_capture_content,
            is_beta=self._self_is_beta,
        )
