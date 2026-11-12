# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
"""Instrumentation for the Claude Agent SDK.

Only the public ``claude_agent_sdk.query`` entry point is wrapped; the SDK's
``InternalClient`` is deliberately left alone.

Span model: one ``query()`` stream is one ``invoke_agent`` span. Tool calls
observed in the stream become ``execute_tool`` spans covering the interval
between the ``tool_use`` block and its ``tool_result``. The tool itself runs
inside the ``claude`` CLI, so that interval is observed, not executed, here.

Lifetime: the agent invocation starts on the first read, not when ``query()``
returns its iterator. A stream created in one context and consumed in another
therefore belongs to the consuming context, and the span excludes the time the
caller held the stream without reading it.

Token accounting: the ``ResultMessage`` carries the run totals and is treated
as authoritative. GenAI input tokens include cache reads and cache writes, so
both are summed into ``gen_ai.usage.input_tokens``; the local agent span has no
attribute of its own for them. Usage reported on individual ``AssistantMessage``
objects is summed only as a fallback for runs that end without a
``ResultMessage`` (aborted or closed early), because counting both would double
count.

Message content is captured as the run's message list, one message per
assistant or tool-result message, and only when content capture is enabled. A
string prompt is captured as the invocation's input message; a streamed
(asynchronous iterable) prompt is left to the caller, so instrumentation never
consumes that iterator.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

from wrapt import FunctionWrapper

from opentelemetry.semconv.attributes.error_attributes import ErrorTypeValues
from opentelemetry.util.genai.handler import TelemetryHandler
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


def _usage_of(message: Any) -> dict[str, Any]:
    raw: Any = getattr(message, "usage", None)
    if not isinstance(raw, dict):
        return {}
    return dict(cast("dict[str, Any]", raw))


def _blocks(message: Any) -> list[Any]:
    raw: Any = getattr(message, "content", None)
    if not isinstance(raw, list):
        return []
    return list(cast("list[Any]", raw))


def _input_tokens(usage: dict[str, Any]) -> int | None:
    """Sum the input components GenAI counts, cache traffic included."""
    components = [
        usage.get("input_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("cache_creation_input_tokens"),
    ]
    if all(component is None for component in components):
        return None
    return sum(component or 0 for component in components)


def _result_error(chunk: Any) -> Error:
    """Build the run error a ``ResultMessage`` reports without raising."""
    subtype = getattr(chunk, "subtype", None)
    errors = getattr(chunk, "errors", None)
    message: str | None = None
    if isinstance(errors, list) and errors:
        message = str(errors[0])
    elif isinstance(subtype, str):
        # Older SDK releases carry no per-error messages; the subtype is the
        # only description the result provides. ``result`` is the model output,
        # not an error description, so it is not used here.
        message = subtype
    return Error(
        message=message,
        type=subtype
        if isinstance(subtype, str)
        else ErrorTypeValues.OTHER.value,
    )


class ClaudeAgentStreamWrapper(AsyncStreamWrapper[Any]):
    """Turns a Claude Agent SDK message stream into one invoke_agent span."""

    def __init__(
        self,
        stream: Any,
        handler: TelemetryHandler,
        *,
        prompt: Any,
        request_model: str | None,
        agent_name: str,
    ) -> None:
        # The invocation is created on the first read; see the module docstring.
        super().__init__(stream)
        self._self_handler = handler
        self._self_prompt = prompt
        self._self_request_model = request_model
        self._self_agent_name = agent_name
        self._self_agent: Any = None
        self._self_outputs: list[OutputMessage] = []
        self._self_result_usage: dict[str, Any] | None = None
        self._self_result_error: Error | None = None
        # None until an assistant message reports the component, so a missing
        # component stays missing instead of being exported as a zero.
        self._self_assistant_input: int | None = None
        self._self_assistant_output: int | None = None
        # tool_use id -> ToolInvocation, for tools executed outside this process
        self._self_pending_tools: dict[str, Any] = {}

    async def _advance(self, read: Any) -> Any:
        # Start here rather than at construction so the span is parented by the
        # context that consumes the stream and its duration starts at the read.
        self._start_agent()
        return await super()._advance(read)

    def _start_agent(self) -> None:
        if self._self_agent is not None:
            return
        invocation = self._self_handler.invoke_local_agent(
            request_model=self._self_request_model,
            agent_name=self._self_agent_name,
        )
        # Only a plain string is captured: an async iterable prompt belongs to
        # the caller and must not be drained just to read it.
        if (
            isinstance(self._self_prompt, str)
            and invocation.should_capture_content
        ):
            invocation.input_messages = [
                InputMessage(
                    role="user",
                    parts=[TextPart(content=self._self_prompt)],
                )
            ]
        # Hand control back to the caller between reads so unrelated caller work
        # is not parented under the agent span.
        invocation.suspend()
        self._self_agent = invocation
        self._self_invocation = invocation

    def _process_chunk(self, chunk: Any) -> None:
        if self._self_agent is None:
            return
        # Newer util releases scope the invocation around this hook; older ones
        # do not, and there tool spans would be created without a parent.
        # Activating here is harmless when the base already did it.
        with self._self_agent.activate():
            self._handle_chunk(chunk)

    def _handle_chunk(self, chunk: Any) -> None:
        kind = type(chunk).__name__
        if kind == "AssistantMessage":
            self._handle_assistant(chunk)
        elif kind == "UserMessage":
            self._handle_user(chunk)
        elif kind == "ResultMessage":
            self._handle_result(chunk)

    def _handle_assistant(self, chunk: Any) -> None:
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

        capture = bool(self._self_agent.should_capture_content)
        parts: list[Any] = []
        for block in _blocks(chunk):
            kind = type(block).__name__
            if kind == "TextBlock":
                text = getattr(block, "text", None)
                if text and capture:
                    parts.append(TextPart(content=text))
            elif kind == "ThinkingBlock":
                thinking = getattr(block, "thinking", None)
                if thinking and capture:
                    parts.append(ReasoningPart(content=thinking))
            elif kind == "ToolUseBlock":
                if capture:
                    parts.append(
                        ToolCallRequestPart(
                            arguments=getattr(block, "input", None),
                            name=getattr(block, "name", None) or "unknown",
                            id=getattr(block, "id", None),
                        )
                    )
                self._start_tool(block)
        # One OutputMessage per SDK message keeps the turn boundaries the model
        # saw, instead of merging every message into a single text blob.
        if parts:
            self._self_outputs.append(
                OutputMessage(role="assistant", parts=parts)
            )

    def _handle_user(self, chunk: Any) -> None:
        capture = bool(self._self_agent.should_capture_content)
        parts: list[Any] = []
        for block in _blocks(chunk):
            if type(block).__name__ != "ToolResultBlock":
                continue
            tool_id = getattr(block, "tool_use_id", None)
            if capture:
                parts.append(
                    ToolCallResponsePart(
                        response=getattr(block, "content", None),
                        id=tool_id if isinstance(tool_id, str) else None,
                    )
                )
            self._end_tool(block)
        if parts:
            self._self_outputs.append(OutputMessage(role="tool", parts=parts))

    def _handle_result(self, chunk: Any) -> None:
        usage = _usage_of(chunk)
        if usage:
            self._self_result_usage = usage
        stop_reason = getattr(chunk, "stop_reason", None)
        if stop_reason:
            self._self_agent.finish_reasons = [stop_reason]
        # The SDK reports a failed run through ``is_error`` without raising, so
        # the run error has to be kept for finalization.
        if getattr(chunk, "is_error", False):
            self._self_result_error = _result_error(chunk)

    def _start_tool(self, block: Any) -> None:
        tool_id = getattr(block, "id", None)
        name = getattr(block, "name", None) or "unknown"
        if not isinstance(tool_id, str) or not tool_id:
            return
        # Observed interval only: the tool runs inside the CLI, so the span
        # covers tool_use -> tool_result, not the tool's own execution time.
        invocation = self._self_handler.tool(
            name=name,
            tool_type="function",
            agent_name=self._self_agent_name,
        )
        invocation.tool_call_id = tool_id
        if invocation.should_capture_content:
            invocation.arguments = getattr(block, "input", None)
        # Suspend right away: a second tool call in the same assistant message
        # must become a sibling under the agent, not a child of the first tool.
        invocation.suspend()
        self._self_pending_tools[tool_id] = invocation

    def _end_tool(self, block: Any) -> None:
        tool_id = getattr(block, "tool_use_id", None)
        if not isinstance(tool_id, str):
            return
        invocation = self._self_pending_tools.pop(tool_id, None)
        if invocation is None:
            return
        content = getattr(block, "content", None)
        if invocation.should_capture_content:
            invocation.tool_result = content
        if getattr(block, "is_error", None):
            # A tool-result error flag is not a Python exception: reporting one
            # would fabricate exception.type and a stack trace that never existed.
            invocation.fail(
                Error(
                    message=content if isinstance(content, str) else None,
                    type=ErrorTypeValues.OTHER.value,
                )
            )
        else:
            invocation.stop()

    def _finalize_pending_tools(
        self, error: BaseException | None = None
    ) -> None:
        """Finish tool spans whose result never arrived in the stream.

        A stream failure is the observed failure of the pending tool, so it is
        reported with that cause. An ordinary end or caller close observes no
        tool error, so those spans end UNSET rather than as failures.
        """
        while self._self_pending_tools:
            _, invocation = self._self_pending_tools.popitem()
            if error is not None:
                invocation.fail(error)
            else:
                invocation.stop()

    def _apply_usage(self) -> None:
        invocation = self._self_agent
        if self._self_result_usage is not None:
            usage = self._self_result_usage
            invocation.input_tokens = _input_tokens(usage)
            invocation.output_tokens = usage.get("output_tokens")
            return
        if self._self_assistant_input is not None or (
            self._self_assistant_output is not None
        ):
            invocation.input_tokens = self._self_assistant_input
            invocation.output_tokens = self._self_assistant_output

    def _apply_output(self) -> None:
        if not self._self_outputs:
            return
        self._self_agent.output_messages = self._self_outputs
        self._self_outputs = []

    def _on_stream_end(self) -> None:
        if self._self_agent is None:
            return
        self._apply_usage()
        self._apply_output()
        self._finalize_pending_tools()
        if self._self_result_error is not None:
            self._self_agent.fail(self._self_result_error)
        else:
            self._self_agent.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        if self._self_agent is None:
            return
        self._apply_usage()
        self._apply_output()
        self._finalize_pending_tools(error)
        self._self_agent.fail(error)


def _wrap_query(handler: TelemetryHandler) -> Any:
    def wrapper(
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        stream = wrapped(*args, **kwargs)
        options = kwargs.get("options")
        return ClaudeAgentStreamWrapper(
            stream,
            handler,
            prompt=kwargs.get("prompt"),
            request_model=getattr(options, "model", None),
            agent_name=getattr(options, "agent_name", None)
            or DEFAULT_AGENT_NAME,
        )

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
