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
from opentelemetry.util.genai.types import OutputMessage

from .messages_extractors import set_invocation_response_attributes
from .utils import (
    StreamBlockState,
    create_stream_block_state,
    stream_block_state_to_part,
    update_stream_block_state,
)

_logger = logging.getLogger(__name__)

try:
    from anthropic.lib.streaming._messages import (  # pylint: disable=no-name-in-module
        accumulate_event as _sdk_accumulate_event,
    )
except ImportError:
    _sdk_accumulate_event = None

if TYPE_CHECKING:
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

_accumulate_takes_json_bufs = False
if accumulate_event is not None:
    try:
        _accumulate_takes_json_bufs = (
            "json_bufs" in inspect.signature(accumulate_event).parameters
        )
    except (ValueError, TypeError):
        _accumulate_takes_json_bufs = False

_accumulation_disabled = False


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
    _self_json_bufs: dict[int, bytes]
    _self_block_states: dict[int, StreamBlockState]
    _self_seen_message_stop: bool
    _self_stop_reason: str | None
    _self_output_tokens: int | None

    def _is_incomplete_response(self) -> bool:
        stop_reason = self._self_stop_reason
        if stop_reason is None and self._self_message is not None:
            raw_stop_reason = getattr(self._self_message, "stop_reason", None)
            stop_reason = (
                raw_stop_reason if isinstance(raw_stop_reason, str) else None
            )
        return (
            self._self_message is not None
            and not self._self_seen_message_stop
            and stop_reason is None
        )

    def _partial_output_messages(self) -> list[OutputMessage] | None:
        if not self._self_block_states:
            return None

        parts = [
            part
            for _, state in sorted(self._self_block_states.items())
            if (part := stream_block_state_to_part(state)) is not None
        ]
        if not parts:
            return None

        role = getattr(self._self_message, "role", "assistant")
        return [
            OutputMessage(
                role=role if isinstance(role, str) else "assistant",
                parts=parts,
                finish_reason="error",
            )
        ]

    def _stop(self) -> None:
        if self._self_message_telemetry_finalized:
            return
        # text_stream and the get_final_* helpers bypass _process_chunk, so the
        # snapshot can be the only record of the response.
        self._adopt_sdk_snapshot()
        incomplete_response = self._is_incomplete_response()
        if self._self_message is not None:
            usage = getattr(self._self_message, "usage", None)
            if usage is not None and (
                incomplete_response or self._self_output_tokens is not None
            ):
                usage.output_tokens = self._self_output_tokens
        _set_response_attributes(
            self._self_invocation,
            self._self_message,
            self._self_capture_content and not incomplete_response,
        )
        if incomplete_response:
            self._self_invocation.finish_reasons = ["error"]
            if (
                self._self_capture_content
                and (output_messages := self._partial_output_messages())
                is not None
            ):
                self._self_invocation.output_messages = output_messages
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
        chunk: RawMessageStreamEvent
        | ParsedMessageStreamEvent[ResponseFormatT],
    ) -> None:
        """Accumulate a final message snapshot from a streaming chunk."""
        global _accumulation_disabled
        self._track_partial_response(chunk)
        if self._adopt_sdk_snapshot():
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

    def _track_partial_response(
        self,
        chunk: RawMessageStreamEvent
        | ParsedMessageStreamEvent[ResponseFormatT],
    ) -> None:
        chunk_type = getattr(chunk, "type", None)
        if chunk_type == "content_block_start":
            index = getattr(chunk, "index", None)
            content_block = getattr(chunk, "content_block", None)
            if isinstance(index, int) and content_block is not None:
                self._self_block_states[index] = create_stream_block_state(
                    content_block
                )
            return

        if chunk_type == "content_block_delta":
            index = getattr(chunk, "index", None)
            delta = getattr(chunk, "delta", None)
            state = (
                self._self_block_states.get(index)
                if isinstance(index, int)
                else None
            )
            if state is not None and delta is not None:
                update_stream_block_state(state, delta)
            return

        if chunk_type == "message_delta":
            delta = getattr(chunk, "delta", None)
            if delta is not None:
                stop_reason = getattr(delta, "stop_reason", None)
                if isinstance(stop_reason, str):
                    self._self_stop_reason = stop_reason
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                output_tokens = getattr(usage, "output_tokens", None)
                if isinstance(output_tokens, int):
                    self._self_output_tokens = output_tokens
            return

        if chunk_type == "message_stop":
            self._self_seen_message_stop = True


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
        self._self_json_bufs = {}
        self._self_block_states = {}
        self._self_seen_message_stop = False
        self._self_stop_reason = None
        self._self_output_tokens = None

    @property
    def response(self) -> _http_lib.Response:
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
        self._self_json_bufs = {}
        self._self_block_states = {}
        self._self_seen_message_stop = False
        self._self_stop_reason = None
        self._self_output_tokens = None

    @property
    def response(self) -> _http_lib.Response:
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
