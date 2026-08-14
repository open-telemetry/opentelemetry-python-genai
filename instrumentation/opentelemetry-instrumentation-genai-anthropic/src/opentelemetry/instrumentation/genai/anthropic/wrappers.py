# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Generic,
    Protocol,
    TypeVar,
    cast,
)

from opentelemetry.util.genai.stream import (
    AsyncStreamManagerWrapper,
    AsyncStreamWrapper,
    SyncStreamManagerWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
)

from .messages_extractors import set_invocation_response_attributes

try:
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        accumulate_event as _sdk_accumulate_event,
    )
except ImportError:
    _sdk_accumulate_event = None

if TYPE_CHECKING:
    import httpx
    from anthropic._streaming import AsyncStream, Stream
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
    from anthropic.types.parsed_message import ParsedMessage

    from opentelemetry.util.genai.invocation import InferenceInvocation


ResponseFormatT = TypeVar("ResponseFormatT")
accumulate_event = cast("Callable[..., Message] | None", _sdk_accumulate_event)


class _StreamWrapperWithStream(Protocol):
    @property
    def stream(self) -> object: ...


def _set_response_attributes(
    invocation: InferenceInvocation,
    result: Message | None,
    capture_content: bool,
) -> None:
    set_invocation_response_attributes(invocation, result, capture_content)


class MessageWrapper:
    """Wrapper for non-streaming Message response that handles telemetry."""

    def __init__(self, message: Message, capture_content: bool):
        self._message = message
        self._capture_content = capture_content

    def extract_into(self, invocation: InferenceInvocation) -> None:
        """Extract response data into the invocation."""
        set_invocation_response_attributes(
            invocation, self._message, self._capture_content
        )

    @property
    def message(self) -> Message:
        """Return the wrapped Message object."""
        return self._message


class _MessagesStreamMixin(Generic[ResponseFormatT]):
    _self_invocation: InferenceInvocation
    _self_message: Message | ParsedMessage[ResponseFormatT] | None
    _self_capture_content: bool
    _self_message_telemetry_finalized: bool

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
        chunk: RawMessageStreamEvent
        | ParsedMessageStreamEvent[ResponseFormatT],
    ) -> None:
        """Accumulate a final message snapshot from a streaming chunk."""
        stream = cast(_StreamWrapperWithStream, self).stream
        snapshot = cast(
            "ParsedMessage[ResponseFormatT] | None",
            getattr(stream, "current_message_snapshot", None),
        )
        if snapshot is not None:
            self._self_message = snapshot
            return
        if accumulate_event is None:
            return
        self._self_message = accumulate_event(
            event=cast("RawMessageStreamEvent", chunk),
            current_snapshot=cast(
                "ParsedMessage[ResponseFormatT] | None", self._self_message
            ),
        )


class MessagesStreamWrapper(
    _MessagesStreamMixin[ResponseFormatT],
    SyncStreamWrapper[
        "RawMessageStreamEvent | ParsedMessageStreamEvent[ResponseFormatT]"
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for Anthropic Stream that handles telemetry."""

    def __init__(
        self,
        stream: Stream[RawMessageStreamEvent] | MessageStream[ResponseFormatT],
        invocation: InferenceInvocation,
        capture_content: bool,
    ):
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_message = None
        self._self_capture_content = capture_content
        self._self_message_telemetry_finalized = False

    @property
    def response(self) -> httpx.Response:
        return finalize_on_close(self.stream.response, self._stop)

    @property
    def stream(
        self,
    ) -> Stream[RawMessageStreamEvent] | MessageStream[ResponseFormatT]:
        return self._self_stream

    @stream.setter
    def stream(
        self,
        stream: Stream[RawMessageStreamEvent] | MessageStream[ResponseFormatT],
    ) -> None:
        self._set_stream(stream)


class AsyncMessagesStreamWrapper(
    _MessagesStreamMixin[ResponseFormatT],
    AsyncStreamWrapper[
        "RawMessageStreamEvent | ParsedMessageStreamEvent[ResponseFormatT]"
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for async Anthropic Stream that handles telemetry."""

    def __init__(
        self,
        stream: AsyncStream[RawMessageStreamEvent]
        | AsyncMessageStream[ResponseFormatT],
        invocation: InferenceInvocation,
        capture_content: bool,
    ):
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_message = None
        self._self_capture_content = capture_content
        self._self_message_telemetry_finalized = False

    @property
    def response(self) -> httpx.Response:
        return finalize_on_aclose(self.stream.response, self._stop)

    @property
    def stream(
        self,
    ) -> (
        AsyncStream[RawMessageStreamEvent]
        | AsyncMessageStream[ResponseFormatT]
    ):
        return self._self_stream

    @stream.setter
    def stream(
        self,
        stream: AsyncStream[RawMessageStreamEvent]
        | AsyncMessageStream[ResponseFormatT],
    ) -> None:
        self._set_stream(stream)


class MessagesStreamManagerWrapper(
    SyncStreamManagerWrapper[
        "MessageStream[ResponseFormatT]",
        "InferenceInvocation",
        "MessagesStreamWrapper[ResponseFormatT]",
    ],
    Generic[ResponseFormatT],
):
    """Wrapper for sync Anthropic stream managers."""

    def __init__(
        self,
        manager: MessageStreamManager[ResponseFormatT],
        invocation_factory: Callable[[], InferenceInvocation],
        capture_content: bool,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content

    def _wrap_stream(
        self,
        stream: MessageStream[ResponseFormatT],
        invocation: InferenceInvocation,
    ) -> MessagesStreamWrapper[ResponseFormatT]:
        return MessagesStreamWrapper(
            stream, invocation, self._self_capture_content
        )


class AsyncMessagesStreamManagerWrapper(
    AsyncStreamManagerWrapper[
        "AsyncMessageStream[ResponseFormatT]",
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
        manager: AsyncMessageStreamManager[ResponseFormatT],
        invocation_factory: Callable[[], InferenceInvocation],
        capture_content: bool,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content

    def _wrap_stream(
        self,
        stream: AsyncMessageStream[ResponseFormatT],
        invocation: InferenceInvocation,
    ) -> AsyncMessagesStreamWrapper[ResponseFormatT]:
        return AsyncMessagesStreamWrapper(
            stream, invocation, self._self_capture_content
        )
