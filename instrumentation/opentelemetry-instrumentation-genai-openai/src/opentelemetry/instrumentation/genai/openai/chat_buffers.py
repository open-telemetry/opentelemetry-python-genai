# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


class ToolCallBuffer:
    def __init__(
        self,
        index: int,
        tool_call_id: str | None,
        function_name: str | None,
    ) -> None:
        self.index: int = index
        self.function_name: str | None = function_name
        self.tool_call_id: str | None = tool_call_id
        self.arguments: list[str] = []

    def append_arguments(self, arguments: str | None) -> None:
        if arguments is not None:
            self.arguments.append(arguments)


class ChoiceBuffer:
    def __init__(self, index: int) -> None:
        self.index: int = index
        self.finish_reason: str | None = None
        self.text_content: list[str] = []
        self.refusal: list[str] = []
        self.tool_calls_buffers: list[ToolCallBuffer | None] = []

    def append_text_content(self, content: str) -> None:
        self.text_content.append(content)

    def append_refusal(self, refusal: str) -> None:
        self.refusal.append(refusal)

    def append_tool_call(self, tool_call: ChoiceDeltaToolCall) -> None:
        function = tool_call.function
        idx = getattr(tool_call, "index", None)
        if idx is None:
            # Some OpenAI-compatible servers omit the index: a delta whose id
            # was seen continues that call, any other delta that carries an id
            # or a function name starts a new call, and one with neither
            # continues the last call.
            idx = next(
                (
                    i
                    for i, buffer in enumerate(self.tool_calls_buffers)
                    if tool_call.id
                    and buffer is not None
                    and buffer.tool_call_id == tool_call.id
                ),
                None,
            )
            if idx is None:
                starts_call = tool_call.id or (function and function.name)
                if starts_call or not self.tool_calls_buffers:
                    idx = len(self.tool_calls_buffers)
                else:
                    idx = len(self.tool_calls_buffers) - 1

        for _ in range(len(self.tool_calls_buffers), idx + 1):
            self.tool_calls_buffers.append(None)

        buffer = self.tool_calls_buffers[idx]
        if buffer is None:
            buffer = ToolCallBuffer(
                idx,
                tool_call.id,
                function.name if function else None,
            )
            self.tool_calls_buffers[idx] = buffer

        if function:
            buffer.append_arguments(function.arguments)
