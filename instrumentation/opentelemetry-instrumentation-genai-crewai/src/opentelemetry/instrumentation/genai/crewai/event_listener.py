# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate CrewAI LLM events into GenAI inference telemetry."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast
from urllib.parse import urlparse

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)

from opentelemetry.instrumentation.utils import is_instrumentation_enabled
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolDefinition,
)


def _event_key(event: Any) -> str | None:
    event_id = getattr(event, "started_event_id", None) or getattr(
        event, "event_id", None
    )
    return str(event_id) if event_id else None


def _tool_call_part(value: Mapping[str, Any]) -> ToolCallRequest | None:
    function = value.get("function")
    if isinstance(function, Mapping):
        function = cast(Mapping[str, Any], function)
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = value.get("name")
        arguments = value.get("arguments", value.get("args"))
    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    call_id = value.get("id")
    return ToolCallRequest(
        name=str(name),
        id=str(call_id) if call_id else None,
        arguments=arguments,
    )


def _message_parts(message: Mapping[str, Any]) -> list[MessagePart]:
    parts: list[MessagePart] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(Text(content=content))
    elif isinstance(content, Sequence):
        for block in cast(Sequence[Any], content):
            if not isinstance(block, Mapping):
                continue
            block = cast(Mapping[str, Any], block)
            text = block.get("text")
            if isinstance(text, str):
                parts.append(Text(content=text))

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, str):
        for tool_call in cast(Sequence[Any], tool_calls):
            if isinstance(tool_call, Mapping) and (
                part := _tool_call_part(cast(Mapping[str, Any], tool_call))
            ):
                parts.append(part)
    return parts


def _input_messages(value: Any) -> list[InputMessage]:
    if isinstance(value, str):
        return [InputMessage(role="user", parts=[Text(content=value)])]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, Sequence):
        return []
    messages: list[InputMessage] = []
    for message in cast(Sequence[Any], value):
        if not isinstance(message, Mapping):
            continue
        message = cast(Mapping[str, Any], message)
        parts = _message_parts(message)
        if parts:
            messages.append(
                InputMessage(
                    role=str(message.get("role") or "user"), parts=parts
                )
            )
    return messages


def _output_messages(
    value: Any, finish_reason: str | None
) -> list[OutputMessage]:
    if isinstance(value, str):
        value = {"role": "assistant", "content": value}
    elif not isinstance(value, (Mapping, Sequence)):
        content = getattr(value, "content", None)
        if isinstance(content, str):
            value = {"role": "assistant", "content": content}

    inputs = _input_messages(value)
    return [
        OutputMessage(
            role=message.role,
            parts=message.parts,
            finish_reason=finish_reason or "",
        )
        for message in inputs
    ]


def _tool_definitions(value: Any) -> list[ToolDefinition] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    definitions: list[ToolDefinition] = []
    for tool in cast(Sequence[Any], value):
        if not isinstance(tool, Mapping):
            continue
        tool = cast(Mapping[str, Any], tool)
        function = tool.get("function")
        definition = (
            cast(Mapping[str, Any], function)
            if isinstance(function, Mapping)
            else tool
        )
        name = definition.get("name")
        if not name:
            continue
        definitions.append(
            FunctionToolDefinition(
                name=str(name),
                description=(
                    str(definition["description"])
                    if definition.get("description") is not None
                    else None
                ),
                parameters=definition.get("parameters", {}),
            )
        )
    return definitions or None


def _server(source: Any) -> tuple[str | None, int | None]:
    endpoint = getattr(source, "base_url", None) or getattr(
        source, "api_base", None
    )
    if not isinstance(endpoint, str):
        return None, None
    parsed = urlparse(endpoint if "://" in endpoint else f"//{endpoint}")
    try:
        return parsed.hostname, parsed.port
    except ValueError:
        return None, None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _set_request_attributes(
    invocation: InferenceInvocation, source: Any, event: LLMCallStartedEvent
) -> None:
    invocation.input_messages = _input_messages(
        getattr(event, "messages", None)
    )
    invocation.tool_definitions = _tool_definitions(
        getattr(event, "tools", None)
    )
    invocation.temperature = _first_not_none(
        getattr(event, "temperature", None),
        getattr(source, "temperature", None),
    )
    invocation.top_p = _first_not_none(
        getattr(event, "top_p", None), getattr(source, "top_p", None)
    )
    invocation.frequency_penalty = _first_not_none(
        getattr(event, "frequency_penalty", None),
        getattr(source, "frequency_penalty", None),
    )
    invocation.presence_penalty = _first_not_none(
        getattr(event, "presence_penalty", None),
        getattr(source, "presence_penalty", None),
    )
    invocation.max_tokens = _int_or_none(
        _first_not_none(
            getattr(event, "max_tokens", None),
            getattr(source, "max_tokens", None),
            getattr(source, "max_completion_tokens", None),
        )
    )
    invocation.stop_sequences = _first_not_none(
        getattr(event, "stop_sequences", None),
        getattr(source, "stop_sequences", None),
    )
    invocation.seed = _first_not_none(
        getattr(event, "seed", None), getattr(source, "seed", None)
    )
    invocation.request_choice_count = _first_not_none(
        getattr(event, "n", None), getattr(source, "n", None)
    )


def _set_usage(invocation: InferenceInvocation, usage: Any) -> None:
    if not isinstance(usage, Mapping):
        return
    usage = cast(Mapping[str, Any], usage)
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    cached_tokens = usage.get(
        "cached_tokens", usage.get("cached_prompt_tokens")
    )
    if input_tokens is not None:
        invocation.input_tokens = _int_or_none(input_tokens)
    if output_tokens is not None:
        invocation.output_tokens = _int_or_none(output_tokens)
    if cached_tokens is not None:
        invocation.cache_read_input_tokens = _int_or_none(cached_tokens)


class CrewAIInferenceEventListener(BaseEventListener):
    """Listen for CrewAI LLM lifecycle events."""

    def __init__(self, telemetry_handler: TelemetryHandler) -> None:
        self._telemetry_handler = telemetry_handler
        self._invocations: dict[str, InferenceInvocation] = {}
        self._handlers: list[tuple[type[Any], Any]] = []
        self._event_bus: CrewAIEventsBus | None = None
        self._lock = threading.RLock()
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        self._event_bus = crewai_event_bus
        self._register(LLMCallStartedEvent, self._on_started)
        self._register(LLMCallCompletedEvent, self._on_completed)
        self._register(LLMCallFailedEvent, self._on_failed)

    def _register(self, event_type: type[Any], handler: Any) -> None:
        if self._event_bus is None:
            return
        registered = self._event_bus.on(event_type)(handler)
        self._handlers.append((event_type, registered))

    def _on_started(self, source: Any, event: LLMCallStartedEvent) -> None:
        if not is_instrumentation_enabled():
            return
        key = _event_key(event)
        if key is None:
            return
        provider = str(getattr(source, "provider", None) or "unknown")
        model = getattr(event, "model", None) or getattr(source, "model", None)
        server_address, server_port = _server(source)
        invocation = self._telemetry_handler.inference(
            provider=provider,
            request_model=str(model) if model else None,
            server_address=server_address,
            server_port=server_port,
        )
        _set_request_attributes(invocation, source, event)
        with self._lock:
            previous = self._invocations.setdefault(key, invocation)
        if previous is not invocation:
            invocation.stop()

    def _pop(self, event: Any) -> InferenceInvocation | None:
        key = _event_key(event)
        if key is None:
            return None
        with self._lock:
            return self._invocations.pop(key, None)

    def _on_completed(self, source: Any, event: LLMCallCompletedEvent) -> None:
        invocation = self._pop(event)
        if invocation is None:
            return
        finish_reason = getattr(event, "finish_reason", None) or getattr(
            event.response, "finish_reason", None
        )
        invocation.output_messages = _output_messages(
            event.response, str(finish_reason) if finish_reason else None
        )
        finish_reason = str(finish_reason) if finish_reason else None
        invocation.finish_reasons = [finish_reason] if finish_reason else None
        response_id = getattr(event, "response_id", None) or getattr(
            event.response, "id", None
        )
        invocation.response_id = str(response_id) if response_id else None
        response_model = getattr(event.response, "model", None)
        if response_model:
            invocation.response_model_name = str(response_model)
        _set_usage(invocation, getattr(event, "usage", None))
        invocation.stop()

    def _on_failed(self, source: Any, event: LLMCallFailedEvent) -> None:
        invocation = self._pop(event)
        if invocation is None:
            return
        error = getattr(event, "error", None)
        invocation.fail(
            error
            if isinstance(error, BaseException)
            else RuntimeError(str(error or "CrewAI LLM call failed"))
        )

    def shutdown(self) -> None:
        if self._event_bus is not None:
            for event_type, handler in self._handlers:
                self._event_bus.off(event_type, handler)
        with self._lock:
            invocations = list(self._invocations.values())
            self._invocations.clear()
        for invocation in invocations:
            invocation.stop()
        self._handlers.clear()
        self._event_bus = None
