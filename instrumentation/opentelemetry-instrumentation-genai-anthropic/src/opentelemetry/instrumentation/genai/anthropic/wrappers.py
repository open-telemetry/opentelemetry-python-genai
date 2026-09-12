# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
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

from .messages_extractors import set_invocation_response_attributes

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


ResponseFormatT = TypeVar("ResponseFormatT")
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

    def _stop(self) -> None:
        if self._self_message_telemetry_finalized:
            return
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
        self._self_invocation.fail(exc)
        self._self_message_telemetry_finalized = True

    def _on_stream_end(self) -> None:
        self._stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._fail(error)

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
        stream = cast(_StreamWrapperWithStream, self).stream
        snapshot = cast(
            "ParsedMessage[ResponseFormatT] | ParsedBetaMessage[ResponseFormatT] | None",
            getattr(stream, "current_message_snapshot", None),
        )
        if snapshot is not None:
            self._self_message = snapshot
            return
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

    @property
    def response(self) -> _http_lib.Response:
        return finalize_on_close(self.stream.response, self._stop)

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

    @property
    def response(self) -> _http_lib.Response:
        return finalize_on_aclose(self.stream.response, self._stop)

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
        is_beta: bool = False,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content
        self._self_is_beta = is_beta

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
        is_beta: bool = False,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content
        self._self_is_beta = is_beta

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
