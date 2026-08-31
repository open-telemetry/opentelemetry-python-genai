# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream wrappers for Portkey AI instrumentation."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, cast

from opentelemetry.instrumentation.genai.portkey.utils import (
    get_property_value,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
)


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
        self.tool_calls_buffers: list[ToolCallBuffer | None] = []

    def append_text_content(self, content: str) -> None:
        self.text_content.append(content)

    def append_tool_call(self, tool_call: Any) -> None:
        idx = get_property_value(tool_call, "index") or 0
        for _ in range(len(self.tool_calls_buffers), idx + 1):
            self.tool_calls_buffers.append(None)

        function = get_property_value(tool_call, "function")
        call_id = get_property_value(tool_call, "id")
        func_name = get_property_value(function, "name") if function else None
        buffer = self.tool_calls_buffers[idx]
        if buffer is None:
            buffer = ToolCallBuffer(
                idx,
                str(call_id) if call_id is not None else None,
                str(func_name) if func_name is not None else None,
            )
            self.tool_calls_buffers[idx] = buffer
        else:
            if buffer.tool_call_id is None and call_id is not None:
                buffer.tool_call_id = str(call_id)
            if buffer.function_name is None and func_name is not None:
                buffer.function_name = str(func_name)

        if function:
            args = get_property_value(function, "arguments")
            if args is not None:
                buffer.append_arguments(str(args))


class _PortkeyStreamMixin:
    """Stream processing hooks shared by sync and async wrappers."""

    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_choice_buffers: list[ChoiceBuffer]
    _self_response_id: str | None
    _self_prompt_tokens: int | None
    _self_completion_tokens: int | None

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_response_id:
            resp_id = get_property_value(chunk, "id")
            if resp_id:
                self._self_response_id = str(resp_id)

        if not self._self_invocation.response_model_name:
            model = get_property_value(chunk, "model")
            if model:
                self._self_invocation.response_model_name = str(model)

        usage = get_property_value(chunk, "usage")
        if usage is not None:
            prompt_tokens = get_property_value(usage, "prompt_tokens")
            if prompt_tokens is not None:
                self._self_prompt_tokens = int(prompt_tokens)
            completion_tokens = get_property_value(usage, "completion_tokens")
            if completion_tokens is not None:
                self._self_completion_tokens = int(completion_tokens)

        choices = get_property_value(chunk, "choices")
        if choices is not None and isinstance(choices, Iterable):
            for choice in cast(Iterable[Any], choices):
                idx = get_property_value(choice, "index") or 0
                for _ in range(len(self._self_choice_buffers), idx + 1):
                    self._self_choice_buffers.append(
                        ChoiceBuffer(len(self._self_choice_buffers))
                    )

                finish_reason = get_property_value(choice, "finish_reason")
                if finish_reason:
                    self._self_choice_buffers[idx].finish_reason = str(
                        finish_reason
                    )

                delta = get_property_value(choice, "delta")
                if delta is not None:
                    content = get_property_value(delta, "content")
                    if content is not None:
                        self._self_choice_buffers[idx].append_text_content(
                            str(content)
                        )
                    tool_calls = get_property_value(delta, "tool_calls")
                    if tool_calls is not None and isinstance(
                        tool_calls, Iterable
                    ):
                        for tool_call in cast(Iterable[Any], tool_calls):
                            self._self_choice_buffers[idx].append_tool_call(
                                tool_call
                            )

                text = get_property_value(choice, "text")
                if text is not None:
                    self._self_choice_buffers[idx].append_text_content(
                        str(text)
                    )

    def _set_output_messages(self) -> None:
        if not self._self_capture_content:
            return
        output_messages: list[OutputMessage] = []
        for choice in self._self_choice_buffers:
            parts: list[MessagePart] = []
            if choice.text_content:
                parts.append(Text(content="".join(choice.text_content)))
            if choice.tool_calls_buffers:
                tool_calls: list[ToolCallRequest] = []
                for tool_call in filter(None, choice.tool_calls_buffers):
                    arguments = None
                    args_str = "".join(tool_call.arguments)
                    if args_str:
                        try:
                            arguments = json.loads(args_str)
                        except Exception:
                            arguments = args_str
                    tool_calls.append(
                        ToolCallRequest(
                            name=tool_call.function_name or "",
                            id=tool_call.tool_call_id,
                            arguments=arguments,
                        )
                    )
                parts.extend(tool_calls)
            output_messages.append(
                OutputMessage(
                    role="assistant",
                    finish_reason=choice.finish_reason or "stop",
                    parts=parts,
                )
            )
        self._self_invocation.output_messages = output_messages

    def _on_stream_end(self) -> None:
        self._cleanup()

    def _on_stream_error(self, error: BaseException) -> None:
        self._cleanup(error)

    def _cleanup(self, error: BaseException | None = None) -> None:
        self._self_invocation.response_id = self._self_response_id
        self._self_invocation.input_tokens = self._self_prompt_tokens
        self._self_invocation.output_tokens = self._self_completion_tokens
        self._self_invocation.finish_reasons = [
            choice.finish_reason
            for choice in self._self_choice_buffers
            if choice.finish_reason
        ]

        self._set_output_messages()

        if error is not None:
            self._self_invocation.fail(error)
        else:
            self._self_invocation.stop()


class PortkeyStreamWrapper(_PortkeyStreamMixin, SyncStreamWrapper[Any]):
    """Synchronous stream wrapper for Portkey completions."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_choice_buffers = []
        self._self_response_id = None
        self._self_prompt_tokens = None
        self._self_completion_tokens = None


class AsyncPortkeyStreamWrapper(_PortkeyStreamMixin, AsyncStreamWrapper[Any]):
    """Asynchronous stream wrapper for Portkey completions."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_choice_buffers = []
        self._self_response_id = None
        self._self_prompt_tokens = None
        self._self_completion_tokens = None


__all__ = [
    "AsyncPortkeyStreamWrapper",
    "ChoiceBuffer",
    "PortkeyStreamWrapper",
    "ToolCallBuffer",
]
