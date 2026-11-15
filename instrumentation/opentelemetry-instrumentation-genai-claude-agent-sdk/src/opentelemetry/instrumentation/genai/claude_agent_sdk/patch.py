# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
"""Instrumentation for ``claude_agent_sdk.query``."""

from __future__ import annotations

import importlib
import os
from collections.abc import AsyncIterator
from typing import Any, cast

from claude_agent_sdk.types import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from wrapt import FunctionWrapper

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation
from opentelemetry.util.genai.stream import AsyncStreamWrapper
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
)

DEFAULT_AGENT_NAME = "claude-agent-sdk"

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_call",
}


def _finish_reason(stop_reason: str) -> str:
    return _FINISH_REASONS.get(stop_reason, stop_reason)


def _usage_of(message: AssistantMessage | ResultMessage) -> dict[str, Any]:
    # AssistantMessage only carries usage in newer SDK releases.
    usage = getattr(message, "usage", None)
    if isinstance(usage, dict):
        return cast("dict[str, Any]", usage)
    return {}


def _input_tokens(usage: dict[str, Any]) -> int | None:
    components = [
        usage.get("input_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("cache_creation_input_tokens"),
    ]
    if all(component is None for component in components):
        return None
    return sum(component or 0 for component in components)


def _result_error(chunk: ResultMessage, *, capture: bool) -> Error:
    message: str | None = None
    # ``errors`` only exists in newer SDK releases.
    errors = getattr(chunk, "errors", None)
    if capture and isinstance(errors, list) and errors:
        # The per-error text is user data, so it only becomes the status
        # description when content capture is enabled.
        message = str(cast("list[object]", errors)[0])
    else:
        message = chunk.subtype
    return Error(message=message, type=chunk.subtype)


def _native_tracing_enabled(options: ClaudeAgentOptions | None) -> bool:
    env = {**os.environ, **(options.env if options else {})}
    return (
        env.get("CLAUDE_CODE_ENABLE_TELEMETRY") == "1"
        and env.get("CLAUDE_CODE_ENHANCED_TELEMETRY_BETA") == "1"
        and env.get("OTEL_TRACES_EXPORTER", "none") != "none"
    )


class ClaudeAgentStreamWrapper(AsyncStreamWrapper[Message]):
    def __init__(
        self,
        stream: AsyncIterator[Message],
        invocation: LocalAgentInvocation,
    ) -> None:
        super().__init__(stream, invocation)
        self._self_invocation = invocation
        invocation.suspend()
        self._self_outputs: list[OutputMessage] = []
        self._self_result_usage: dict[str, Any] | None = None
        self._self_result_error: Error | None = None
        self._self_sdk_session_seen = False
        self._self_assistant_input: int | None = None
        self._self_assistant_output: int | None = None

    def _process_chunk(self, chunk: Message) -> None:
        if isinstance(chunk, AssistantMessage):
            self._handle_assistant(chunk)
        elif isinstance(chunk, UserMessage):
            self._handle_user(chunk)
        elif isinstance(chunk, ResultMessage):
            self._handle_result(chunk)
        elif isinstance(chunk, SystemMessage):
            self._handle_system(chunk)

    def _set_conversation_id(self, session_id: str) -> None:
        # The first SDK session id wins; the init message and the result agree.
        if self._self_sdk_session_seen:
            return
        self._self_invocation.conversation_id = session_id
        self._self_sdk_session_seen = True

    def _handle_system(self, chunk: SystemMessage) -> None:
        if chunk.subtype != "init":
            return
        session_id = chunk.data.get("session_id")
        if isinstance(session_id, str) and session_id:
            self._set_conversation_id(session_id)

    def _handle_assistant(self, chunk: AssistantMessage) -> None:
        usage = _usage_of(chunk)
        input_tokens = _input_tokens(usage)
        if input_tokens is not None:
            self._self_assistant_input = (
                self._self_assistant_input or 0
            ) + input_tokens
        output_tokens = usage.get("output_tokens")
        if output_tokens is not None:
            self._self_assistant_output = (
                self._self_assistant_output or 0
            ) + output_tokens

        parts: list[Any] = []
        for block in chunk.content:
            if isinstance(block, TextBlock) and block.text:
                if self._self_invocation.should_capture_content:
                    parts.append(TextPart(content=block.text))
            elif isinstance(block, ThinkingBlock) and block.thinking:
                if self._self_invocation.should_capture_content:
                    parts.append(ReasoningPart(content=block.thinking))
            elif isinstance(block, ToolUseBlock):
                if self._self_invocation.should_capture_content:
                    parts.append(
                        ToolCallRequestPart(
                            arguments=block.input,
                            name=block.name,
                            id=block.id,
                        )
                    )
        if parts:
            self._self_outputs.append(
                OutputMessage(role="assistant", parts=parts)
            )

    def _handle_user(self, chunk: UserMessage) -> None:
        parts: list[Any] = []
        if isinstance(chunk.content, list):
            for block in chunk.content:
                if isinstance(block, ToolResultBlock):
                    if self._self_invocation.should_capture_content:
                        parts.append(
                            ToolCallResponsePart(
                                response=block.content,
                                id=block.tool_use_id,
                            )
                        )
        if parts:
            self._self_outputs.append(OutputMessage(role="tool", parts=parts))

    def _handle_result(self, chunk: ResultMessage) -> None:
        if chunk.session_id:
            self._set_conversation_id(chunk.session_id)
        if chunk.usage:
            self._self_result_usage = dict(chunk.usage)
        # ``stop_reason`` only exists in newer SDK releases.
        stop_reason = getattr(chunk, "stop_reason", None)
        if stop_reason:
            self._self_invocation.finish_reasons = [
                _finish_reason(stop_reason)
            ]
        if chunk.is_error:
            self._self_result_error = _result_error(
                chunk,
                capture=self._self_invocation.should_capture_content,
            )

    def _apply_usage(self) -> None:
        # The ResultMessage carries the run totals; AssistantMessage usage is
        # only a fallback for runs that end without one, because counting both
        # would double count.
        if self._self_result_usage is not None:
            usage = self._self_result_usage
            self._self_invocation.input_tokens = _input_tokens(usage)
            self._self_invocation.output_tokens = usage.get("output_tokens")
            return
        if self._self_assistant_input is not None or (
            self._self_assistant_output is not None
        ):
            self._self_invocation.input_tokens = self._self_assistant_input
            self._self_invocation.output_tokens = self._self_assistant_output

    def _apply_output(self) -> None:
        if self._self_outputs:
            self._self_invocation.output_messages = self._self_outputs
            self._self_outputs = []

    def _on_stream_end(self) -> None:
        self._apply_usage()
        self._apply_output()
        if self._self_result_error is not None:
            self._self_invocation.fail(self._self_result_error)
        else:
            self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._apply_usage()
        self._apply_output()
        self._self_invocation.fail(error)


def _wrap_query(handler: TelemetryHandler) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        options = kwargs.get("options")
        if _native_tracing_enabled(options):
            return wrapped(*args, **kwargs)
        invocation = handler.invoke_local_agent(
            request_model=options.model if options else None,
            agent_name=DEFAULT_AGENT_NAME,
        )
        prompt = kwargs.get("prompt")
        if isinstance(prompt, str) and invocation.should_capture_content:
            invocation.input_messages = [
                InputMessage(role="user", parts=[TextPart(content=prompt)])
            ]
        return ClaudeAgentStreamWrapper(wrapped(*args, **kwargs), invocation)

    return wrapper


_PATCHED: list[tuple[Any, str, Any]] = []


def _patch(module_name: str, attr: str, handler: TelemetryHandler) -> None:
    module = importlib.import_module(module_name)
    original = getattr(module, attr)
    if isinstance(original, FunctionWrapper):
        return
    setattr(module, attr, FunctionWrapper(original, _wrap_query(handler)))
    _PATCHED.append((module, attr, original))


def instrument(handler: TelemetryHandler) -> None:
    for module_name in ("claude_agent_sdk.query", "claude_agent_sdk"):
        _patch(module_name, "query", handler)


def uninstrument() -> None:
    while _PATCHED:
        module, attr, original = _PATCHED.pop()
        setattr(module, attr, original)
