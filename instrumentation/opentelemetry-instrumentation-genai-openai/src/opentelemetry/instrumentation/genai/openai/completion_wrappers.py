# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for the legacy completions API (``completions.create``)."""

from __future__ import annotations

import logging

from openai import AsyncStream, Stream
from openai.types import Completion, CompletionUsage

from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
)

from ._raw_response import _http_lib
from .chat_buffers import ChoiceBuffer
from .utils import set_completion_response

_logger = logging.getLogger(__name__)


class _CompletionStreamMixin:
    """Completion-chunk hooks shared by sync and async stream wrappers."""

    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_choice_buffers: list[ChoiceBuffer]
    _self_response_id: str | None
    _self_system_fingerprint: str | None
    _self_usage: CompletionUsage | None

    def _init_state(
        self, invocation: InferenceInvocation, capture_content: bool
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_choice_buffers = []
        self._self_response_id = None
        self._self_system_fingerprint = None
        self._self_usage = None

    def _process_chunk(self, chunk: Completion) -> None:
        if not isinstance(chunk, Completion):
            # raw-response stream can be parsed into a caller-defined chunk type.
            _logger.debug(
                "Skipping telemetry for unrecognized completion chunk type %s",
                type(chunk).__name__,
            )
            return

        # Set eagerly so the per-chunk streaming timing metrics carry
        # gen_ai.response.model.
        if self._self_invocation.response_model_name is None:
            self._self_invocation.response_model_name = chunk.model
        if self._self_response_id is None:
            self._self_response_id = chunk.id
        if chunk.system_fingerprint is not None:
            self._self_system_fingerprint = chunk.system_fingerprint
        if chunk.usage is not None:
            self._self_usage = chunk.usage

        buffers = self._self_choice_buffers
        for choice in chunk.choices:
            for index in range(len(buffers), choice.index + 1):
                buffers.append(ChoiceBuffer(index))
            buffer = buffers[choice.index]
            # Typed as always set, but null in every chunk before the last.
            if choice.finish_reason is not None:
                buffer.finish_reason = choice.finish_reason
            if self._self_capture_content:
                buffer.append_text_content(choice.text)

    def _on_stream_end(self) -> None:
        self._finalize(None)

    def _on_stream_error(self, error: BaseException) -> None:
        self._finalize(error)

    def _finalize(self, error: BaseException | None) -> None:
        buffers = self._self_choice_buffers
        set_completion_response(
            self._self_invocation,
            response_id=self._self_response_id,
            system_fingerprint=self._self_system_fingerprint,
            usage=self._self_usage,
            finish_reasons=[
                buffer.finish_reason
                for buffer in buffers
                if buffer.finish_reason is not None
            ],
            # Buffers are allocated up to the highest index seen, so one with
            # no chunks belongs to a choice that never arrived.
            texts=[
                "".join(buffer.text_content)
                for buffer in buffers
                if buffer.text_content
            ]
            if self._self_capture_content
            else None,
        )
        if error is None:
            self._self_invocation.stop()
        else:
            self._self_invocation.fail(error)


class CompletionStreamWrapper(
    _CompletionStreamMixin,
    SyncStreamWrapper[Completion],
):
    def __init__(
        self,
        stream: Stream[Completion],
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_state(invocation, capture_content)

    @property
    def response(self) -> _http_lib.Response:
        # A caller can close this HTTP response instead of the stream, so
        # that close has to finalize telemetry too.
        return finalize_on_close(
            self._self_stream.response,
            self._finalize_success,
            execution_context=self._execution_context,
        )


class AsyncCompletionStreamWrapper(
    _CompletionStreamMixin,
    AsyncStreamWrapper[Completion],
):
    def __init__(
        self,
        stream: AsyncStream[Completion],
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_state(invocation, capture_content)

    @property
    def response(self) -> _http_lib.Response:
        # See CompletionStreamWrapper.response.
        return finalize_on_aclose(
            self._self_stream.response,
            self._finalize_success,
            execution_context=self._execution_context,
        )


__all__ = [
    "AsyncCompletionStreamWrapper",
    "CompletionStreamWrapper",
]
