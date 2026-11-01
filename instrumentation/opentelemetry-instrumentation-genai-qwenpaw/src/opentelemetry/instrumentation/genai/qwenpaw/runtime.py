# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Observe QwenPaw 2.x's request and SSE envelope boundary."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import cast
from urllib.parse import urlsplit

from opentelemetry.context import Context, attach, detach, get_current
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation
from opentelemetry.util.genai.stream import (
    AbandonedStreamError,
    AsyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    Error,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
)
from opentelemetry.util.genai.utils import image_from_url

from .utils import non_empty_str


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, dict):
        result = cast("dict[str, object]", value).get(name, default)
    else:
        result = getattr(value, name, default)
    return result.value if isinstance(result, Enum) else result


def _items(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(cast("list[object]", value))
    return []


def _json_value(value: object) -> object:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            pass
    return value


def _part(block: object, message_type: object) -> MessagePart | None:
    kind = _field(block, "type")
    text = _field(block, "text")
    if kind == "text" and isinstance(text, str):
        if message_type == "reasoning":
            return ReasoningPart(content=text)
        return TextPart(content=text)
    if kind == "data":
        data = _field(block, "data")
        call_id = non_empty_str(_field(data, "call_id"))
        if message_type in ("function_call", "mcp_tool_call", "plugin_call"):
            name = non_empty_str(_field(data, "name"))
            if name:
                return ToolCallRequestPart(
                    name=name,
                    id=call_id,
                    arguments=_json_value(_field(data, "arguments")),
                )
        if message_type in (
            "function_call_output",
            "mcp_tool_call_output",
            "plugin_call_output",
        ):
            return ToolCallResponsePart(
                id=call_id, response=_field(data, "output")
            )
    uri_field = {
        "image": "image_url",
        "audio": "data",
        "video": "video_url",
        "file": "file_url",
    }.get(str(kind))
    if uri_field:
        uri = _field(block, uri_field)
        if isinstance(uri, str) and uri:
            audio_format = _field(block, "format")
            if (
                kind == "audio"
                and isinstance(audio_format, str)
                and audio_format
                and not urlsplit(uri).scheme
            ):
                uri = f"data:audio/{audio_format};base64,{uri}"
            return image_from_url(
                uri,
                modality="document" if kind == "file" else str(kind),
            )
    return None


def _parts(message: object) -> list[MessagePart]:
    return [
        part
        for block in _items(_field(message, "content"))
        if (part := _part(block, _field(message, "type"))) is not None
    ]


def input_messages_from_request(request: object) -> list[InputMessage]:
    """Read envelope messages without normalizing or mutating the request."""
    return [
        InputMessage(role=str(_field(message, "role") or "user"), parts=parts)
        for message in _items(_field(request, "input"))
        if (parts := _parts(message))
    ]


_current_runtime: ContextVar[_RuntimeInvocation | None] = ContextVar(
    "qwenpaw_runtime_invocation", default=None
)


class _RuntimeInvocation:
    """Bind the invocation after the SDK lazily normalizes its request."""

    _suspended = True

    def __init__(
        self, handler: TelemetryHandler, runtime: object, request: object
    ) -> None:
        self.handler = handler
        self.runtime = runtime
        self.request = request
        self.invocation: LocalAgentInvocation | None = None
        self._context: Context | None = None

    def start(self, request: object) -> LocalAgentInvocation:
        if self.invocation is None:
            workspace = getattr(self.runtime, "workspace", None)
            # Workspace.config can load from disk; use only the cached configuration.
            config = getattr(workspace, "_config", None)
            self.invocation = self.handler.invoke_local_agent(
                agent_name=non_empty_str(getattr(config, "name", None)),
                conversation_id=non_empty_str(_field(request, "session_id")),
            )
            if self.invocation.should_capture_content:
                self.invocation.input_messages = input_messages_from_request(
                    request
                )
        return self.invocation

    @contextmanager
    def activate(self) -> Iterator[None]:
        token = _current_runtime.set(self)
        context_token = (
            attach(self._context) if self._context is not None else None
        )
        try:
            yield
        finally:
            # Runtime hooks can keep a child span open across SSE yields.
            self._context = get_current()
            if self.invocation is not None:
                self.invocation.suspend()
            if context_token is not None:
                detach(context_token)
            _current_runtime.reset(token)


def observe_context_build(
    wrapped: Callable[..., object],
    instance: object,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    current = _current_runtime.get()
    if current is not None:
        current.start(args[0] if args else kwargs.get("request"))
    return wrapped(*args, **kwargs)


class RuntimeStreamWrapper(AsyncStreamWrapper[object]):
    """Track envelope snapshots and deltas while preserving the lazy stream."""

    def __init__(
        self,
        stream: AsyncGenerator[object, None],
        pending: _RuntimeInvocation,
    ) -> None:
        super().__init__(stream, pending)
        self._self_pending = pending
        self._self_messages: dict[str, OutputMessage] = {}
        self._self_blocks: dict[str, dict[object, MessagePart]] = {}
        self._self_message_types: dict[str, object] = {}
        self._self_message_roles: dict[str, str] = {}
        self._self_tool_arguments: dict[str, str] = {}
        self._self_complete = False
        self._self_error: Error | None = None

    def _process_chunk(self, chunk: object) -> None:
        invocation = self._self_pending.start(self._self_pending.request)
        if _field(chunk, "object") == "response":
            session_id = non_empty_str(_field(chunk, "session_id"))
            if session_id:
                invocation.conversation_id = session_id
            status = _field(chunk, "status")
            self._self_complete = status == "completed"
            if status in ("failed", "cancelled"):
                error = _field(chunk, "error")
                self._self_error = Error(
                    type=non_empty_str(_field(error, "code")) or str(status),
                    message=non_empty_str(_field(error, "message")),
                )
            if invocation.should_capture_content:
                for message in _items(_field(chunk, "output")):
                    self._capture_message(message)
        elif invocation.should_capture_content:
            if _field(chunk, "object") == "message":
                self._capture_message(chunk)
            elif _field(chunk, "object") == "content":
                self._capture_content(chunk)

    def _capture_message(self, message: object) -> None:
        role = _field(message, "role")
        if role not in ("assistant", "tool"):
            return
        message_id = non_empty_str(_field(message, "id"))
        if message_id is None:
            return
        blocks = self._self_blocks.setdefault(message_id, {})
        for position, block in enumerate(_items(_field(message, "content"))):
            part = _part(block, _field(message, "type"))
            if part is not None:
                index = _field(block, "index")
                blocks[position if index is None else index] = part
        if blocks:
            self._self_messages[message_id] = OutputMessage(
                role=str(role), parts=list(blocks.values())
            )
        self._self_message_types[message_id] = _field(message, "type")
        self._self_message_roles[message_id] = str(role)

    def _capture_content(self, content: object) -> None:
        message_id = non_empty_str(_field(content, "msg_id"))
        if message_id is None:
            return
        blocks = self._self_blocks.setdefault(message_id, {})
        index = _field(content, "index", 0)
        previous = blocks.get(index)
        part = _part(content, self._self_message_types.get(message_id))
        if (
            isinstance(previous, ToolCallRequestPart)
            and _field(content, "delta") is True
        ):
            delta = _field(_field(content, "data"), "arguments")
            if isinstance(delta, str):
                arguments = (
                    self._self_tool_arguments.get(message_id, "") + delta
                )
                self._self_tool_arguments[message_id] = arguments
                part = ToolCallRequestPart(
                    name=previous.name,
                    id=previous.id,
                    arguments=_json_value(arguments),
                )
        elif isinstance(part, ToolCallRequestPart):
            arguments = _field(_field(content, "data"), "arguments")
            if isinstance(arguments, str):
                self._self_tool_arguments[message_id] = arguments
        if part is None:
            return
        if (
            _field(content, "delta") is True
            and isinstance(part, (TextPart, ReasoningPart))
            and isinstance(previous, (TextPart, ReasoningPart))
            and part.type == previous.type
        ):
            part = type(part)(content=previous.content + part.content)
        blocks[index] = part
        self._self_messages[message_id] = OutputMessage(
            role=self._self_message_roles.get(message_id, "assistant"),
            parts=list(blocks.values()),
        )

    def _on_stream_end(self) -> None:
        invocation = self._self_pending.invocation
        if invocation is None:
            return
        invocation.output_messages = list(self._self_messages.values())
        if self._self_error is not None:
            invocation.fail(self._self_error)
        else:
            if self._self_complete:
                invocation.finish_reasons = ["stop"]
            invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        if self._self_pending.invocation is None and isinstance(
            error, AbandonedStreamError
        ):
            return
        invocation = self._self_pending.start(self._self_pending.request)
        invocation.output_messages = list(self._self_messages.values())
        invocation.fail(error)


def make_runtime_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., RuntimeStreamWrapper]:
    """Build the wrapt wrapper for ``Runtime.run(request)``."""

    def runtime_wrapper(
        wrapped: Callable[..., AsyncGenerator[object, None]],
        instance: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> RuntimeStreamWrapper:
        request = args[0] if args else kwargs.get("request")
        pending = _RuntimeInvocation(handler, instance, request)
        try:
            stream = wrapped(*args, **kwargs)
        except BaseException as error:
            pending.start(request).fail(error)
            raise
        return RuntimeStreamWrapper(stream, pending)

    return runtime_wrapper
