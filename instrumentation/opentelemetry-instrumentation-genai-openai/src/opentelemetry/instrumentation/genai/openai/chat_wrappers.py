# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from types import TracebackType

from openai import AsyncStream, Stream
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from wrapt import ObjectProxy

from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
)
from opentelemetry.util.genai.types import (
    OutputMessage,
    Role,
    TextPart,
    ToolCallRequestPart,
)

from ._raw_response import _http_lib
from .chat_buffers import ChoiceBuffer
from .utils import (
    map_finish_reason,
    set_chat_usage,
)

_logger = logging.getLogger(__name__)

# Collects the stream wrappers the patched create() builds while a
# chat.completions.stream() manager is being entered.
_entered_chat_streams: ContextVar[
    list[ChatStreamWrapper | AsyncChatStreamWrapper] | None
] = ContextVar("_entered_chat_streams", default=None)


def _register_entered_chat_stream(
    stream_wrapper: ChatStreamWrapper | AsyncChatStreamWrapper,
) -> None:
    entered = _entered_chat_streams.get()
    if entered is not None:
        entered.append(stream_wrapper)


class _ChatStreamMixin:
    """Chat-specific hooks shared by sync and async stream wrappers."""

    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_choice_buffers: list[ChoiceBuffer]
    _self_response_id: str | None
    _self_service_tier: str | None
    _self_usage: CompletionUsage | None

    def _set_response_model(self, chunk: ChatCompletionChunk) -> None:
        # Set eagerly so the per-chunk streaming timing metrics carry
        # gen_ai.response.model.
        if self._self_invocation.response_model_name:
            return

        if chunk.model:
            self._self_invocation.response_model_name = chunk.model

    def _set_response_id(self, chunk: ChatCompletionChunk) -> None:
        if self._self_response_id:
            return

        if chunk.id:
            self._self_response_id = chunk.id

    def _set_response_service_tier(self, chunk: ChatCompletionChunk) -> None:
        if self._self_service_tier:
            return

        service_tier = getattr(chunk, "service_tier", None)
        if service_tier:
            self._self_service_tier = service_tier

    def _build_streaming_response(self, chunk: ChatCompletionChunk) -> None:
        if chunk.choices is None:
            return

        for choice in chunk.choices:
            if not choice.delta:
                continue

            for idx in range(len(self._self_choice_buffers), choice.index + 1):
                self._self_choice_buffers.append(ChoiceBuffer(idx))

            if choice.finish_reason:
                self._self_choice_buffers[
                    choice.index
                ].finish_reason = choice.finish_reason

            if choice.delta.content is not None:
                self._self_choice_buffers[choice.index].append_text_content(
                    choice.delta.content
                )

            refusal = getattr(choice.delta, "refusal", None)
            if isinstance(refusal, str):
                self._self_choice_buffers[choice.index].append_refusal(refusal)

            if choice.delta.tool_calls is not None:
                for tool_call in choice.delta.tool_calls:
                    self._self_choice_buffers[choice.index].append_tool_call(
                        tool_call
                    )

    def _process_chunk(self, chunk: ChatCompletionChunk) -> None:
        if not isinstance(chunk, ChatCompletionChunk):
            # raw-response stream can be parsed into a caller-defined chunk type.
            _logger.debug(
                "Skipping telemetry for unrecognized chat chunk type %s",
                type(chunk).__name__,
            )
            return

        self._set_response_id(chunk)
        self._set_response_model(chunk)
        self._set_response_service_tier(chunk)
        self._build_streaming_response(chunk)
        usage: CompletionUsage | None = chunk.usage
        if usage is not None:
            self._self_usage = usage

    def _set_output_messages(self) -> None:
        if not self._self_capture_content:  # optimization
            return
        output_messages = []
        for choice in self._self_choice_buffers:
            message = OutputMessage(
                role=Role.ASSISTANT.value,
                finish_reason=map_finish_reason(choice.finish_reason),
                parts=[],
            )
            if choice.text_content:
                message.parts.append(
                    TextPart(content="".join(choice.text_content))
                )
            # A refused completion streams its text in `refusal` deltas
            # instead of `content`.
            if refusal := "".join(choice.refusal):
                message.parts.append(TextPart(content=refusal))
            if choice.tool_calls_buffers:
                tool_calls = []
                for tool_call in filter(None, choice.tool_calls_buffers):
                    arguments = None
                    arguments_str = "".join(tool_call.arguments)
                    if arguments_str:
                        try:
                            arguments = json.loads(arguments_str)
                        except json.JSONDecodeError:
                            arguments = arguments_str
                    tool_call_part = ToolCallRequestPart(
                        name=tool_call.function_name,
                        id=tool_call.tool_call_id,
                        arguments=arguments,
                    )
                    tool_calls.append(tool_call_part)
                message.parts.extend(tool_calls)
            output_messages.append(message)

        self._self_invocation.output_messages = output_messages

    def _on_stream_end(self) -> None:
        self._cleanup()

    def _on_stream_error(self, error: BaseException) -> None:
        self._cleanup(error)

    def _cleanup(self, error: BaseException | None = None) -> None:
        self._self_invocation.response_id = self._self_response_id
        if self._self_usage is not None:
            set_chat_usage(
                invocation=self._self_invocation, usage=self._self_usage
            )
        finish_reasons = [
            choice.finish_reason
            for choice in self._self_choice_buffers
            if choice.finish_reason
        ]
        if finish_reasons:
            self._self_invocation.finish_reasons = finish_reasons
        if self._self_service_tier:
            self._self_invocation.attributes.update(
                {
                    OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER: self._self_service_tier
                },
            )

        self._set_output_messages()

        if error:
            self._self_invocation.fail(error)
        else:
            self._self_invocation.stop()


class ChatStreamWrapper(
    _ChatStreamMixin,
    SyncStreamWrapper[ChatCompletionChunk],
):
    def __init__(
        self,
        stream: Stream[ChatCompletionChunk],
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        invocation.suspend()
        self._self_choice_buffers = []
        self._self_capture_content = capture_content
        self._self_response_id = None
        self._self_service_tier = None
        self._self_usage = None
        _register_entered_chat_stream(self)

    @property
    def response(self) -> _http_lib.Response:
        # ChatCompletionStream (chat.completions.stream()) closes this HTTP
        # response directly instead of the stream, so that close has to
        # finalize telemetry too.
        return finalize_on_close(
            self._self_stream.response,
            self._finalize_success,
            execution_context=self._execution_context,
        )


class AsyncChatStreamWrapper(
    _ChatStreamMixin,
    AsyncStreamWrapper[ChatCompletionChunk],
):
    def __init__(
        self,
        stream: AsyncStream[ChatCompletionChunk],
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        invocation.suspend()
        self._self_choice_buffers = []
        self._self_capture_content = capture_content
        self._self_response_id = None
        self._self_service_tier = None
        self._self_usage = None
        _register_entered_chat_stream(self)

    @property
    def response(self) -> _http_lib.Response:
        # See ChatStreamWrapper.response.
        return finalize_on_aclose(
            self._self_stream.response,
            self._finalize_success,
            execution_context=self._execution_context,
        )


class ChatStreamManagerWrapper(ObjectProxy):
    """Wrapper for the manager returned by ``chat.completions.stream()``.

    The SDK manager's ``__exit__`` only closes the stream and drops the
    caller's exception, so it is recorded on the chat stream wrapper first.
    """

    def __init__(self, manager: object) -> None:
        super().__init__(manager)
        self._self_stream_wrapper: ChatStreamWrapper | None = None

    def __enter__(self) -> object:
        entered: list[ChatStreamWrapper | AsyncChatStreamWrapper] = []
        token = _entered_chat_streams.set(entered)
        try:
            stream = self.__wrapped__.__enter__()
        finally:
            _entered_chat_streams.reset(token)
        if entered and isinstance(entered[0], ChatStreamWrapper):
            self._self_stream_wrapper = entered[0]
        return stream

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        stream_wrapper = self._self_stream_wrapper
        self._self_stream_wrapper = None
        if stream_wrapper is not None and exc_val is not None:
            stream_wrapper.__exit__(exc_type, exc_val, exc_tb)
        return self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)


class AsyncChatStreamManagerWrapper(ObjectProxy):
    """Async counterpart of ``ChatStreamManagerWrapper``."""

    def __init__(self, manager: object) -> None:
        super().__init__(manager)
        self._self_stream_wrapper: AsyncChatStreamWrapper | None = None

    async def __aenter__(self) -> object:
        entered: list[ChatStreamWrapper | AsyncChatStreamWrapper] = []
        token = _entered_chat_streams.set(entered)
        try:
            stream = await self.__wrapped__.__aenter__()
        finally:
            _entered_chat_streams.reset(token)
        if entered and isinstance(entered[0], AsyncChatStreamWrapper):
            self._self_stream_wrapper = entered[0]
        return stream

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        stream_wrapper = self._self_stream_wrapper
        self._self_stream_wrapper = None
        if stream_wrapper is not None and exc_val is not None:
            await stream_wrapper.__aexit__(exc_type, exc_val, exc_tb)
        return await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)


__all__ = [
    "AsyncChatStreamManagerWrapper",
    "AsyncChatStreamWrapper",
    "ChatStreamManagerWrapper",
    "ChatStreamWrapper",
]
