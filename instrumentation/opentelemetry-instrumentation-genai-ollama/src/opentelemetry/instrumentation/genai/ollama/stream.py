# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Ollama chat responses."""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.genai.ollama.utils import (
    apply_final_chunk,
    get,
    output_message,
    tool_call_requests,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import MessagePart


class _ChatStreamMixin:
    """Chunk handling shared by the sync and async wrappers."""

    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_content: list[str]
    _self_thinking: list[str]
    _self_tool_calls: list[MessagePart]
    _self_final_chunk: Any

    def _init_buffers(self, capture_content: bool) -> None:
        self._self_capture_content = capture_content
        self._self_content = []
        self._self_thinking = []
        self._self_tool_calls = []
        self._self_final_chunk = None

    def _process_chunk(self, chunk: Any) -> None:
        message = get(chunk, "message")
        if content := get(message, "content"):
            self._self_content.append(str(content))
        if thinking := get(message, "thinking"):
            self._self_thinking.append(str(thinking))
        self._self_tool_calls.extend(
            tool_call_requests(get(message, "tool_calls"))
        )
        # Only the last chunk carries the done reason and token counts.
        if get(chunk, "done"):
            self._self_final_chunk = chunk

    def _record_response(self) -> None:
        invocation = self._self_invocation
        if self._self_final_chunk is not None:
            apply_final_chunk(
                invocation,
                self._self_final_chunk,
                bool(self._self_tool_calls),
            )
        if self._self_capture_content:
            invocation.output_messages = [
                output_message(
                    "".join(self._self_content),
                    "".join(self._self_thinking),
                    self._self_tool_calls,
                )
            ]

    def _on_stream_end(self) -> None:
        self._record_response()
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._record_response()
        self._self_invocation.fail(error)


class ChatStreamWrapper(_ChatStreamMixin, SyncStreamWrapper[Any]):
    """Wraps the generator returned by ``Client.chat(stream=True)``."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        invocation.suspend()
        self._init_buffers(capture_content)


class AsyncChatStreamWrapper(_ChatStreamMixin, AsyncStreamWrapper[Any]):
    """Wraps the async generator returned by ``AsyncClient.chat(stream=True)``."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        invocation.suspend()
        self._init_buffers(capture_content)
