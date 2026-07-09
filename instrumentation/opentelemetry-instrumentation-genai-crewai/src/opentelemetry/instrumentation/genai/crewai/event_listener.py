# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.event_bus import CrewAIEventsBus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
)
from opentelemetry import context as context_api
from opentelemetry.instrumentation.genai.crewai.utils import safe_json_dumps
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _is_suppressed() -> bool:
    return bool(context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY))


def _event_id(event: Any) -> str | None:
    value = getattr(event, "event_id", None)
    return str(value) if value else None


def _started_event_id(event: Any) -> str | None:
    value = getattr(event, "started_event_id", None) or getattr(
        event, "event_id", None
    )
    return str(value) if value else None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _normalize_tool_arguments(value: Any) -> Any:
    """Normalize CrewAI/OpenAI-style function arguments.

    CrewAI can pass function arguments as JSON, or as a quoted JSON string.
    Keep valid JSON strings stable and peel one quote layer when that exposes
    valid JSON, matching the donated OpenInference behavior.
    """
    if not isinstance(value, str):
        return value
    try:
        json.loads(value)
    except json.JSONDecodeError:
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            inner = value[1:-1]
            try:
                json.loads(inner)
            except json.JSONDecodeError:
                return value
            return inner
        return value
    return value


def _message_from_mapping(
    message: Mapping[str, Any],
    default_role: str,
) -> tuple[str, list[MessagePart]]:
    normalized_message = dict(message)
    role = str(normalized_message.get("role") or default_role)
    parts: list[MessagePart] = []

    content = normalized_message.get("content")
    if content is not None:
        parts.append(
            Text(
                content=content
                if isinstance(content, str)
                else safe_json_dumps(content)
            )
        )

    tool_calls = normalized_message.get("tool_calls")
    if isinstance(tool_calls, Sequence) and not isinstance(
        tool_calls, (str, bytes, bytearray)
    ):
        for tool_call in tool_calls:
            if not isinstance(tool_call, Mapping):
                continue
            function = tool_call.get("function")
            name = tool_call.get("name")
            arguments: Any = tool_call.get("args")
            if isinstance(function, Mapping):
                name = function.get("name") or name
                arguments = _normalize_tool_arguments(
                    function.get("arguments", arguments)
                )
            parts.append(
                ToolCallRequest(
                    name=str(name or "unknown"),
                    id=str(tool_call["id"]) if tool_call.get("id") else None,
                    arguments=arguments,
                )
            )

    if not parts:
        parts.append(Text(content=safe_json_dumps(normalized_message)))
    return role, parts


def _input_messages(value: Any, default_role: str = "user") -> list[InputMessage]:
    if value is None:
        return []
    if isinstance(value, str):
        return [
            InputMessage(
                role=default_role,
                parts=cast(list[MessagePart], [Text(content=value)]),
            )
        ]
    if isinstance(value, Mapping):
        role, parts = _message_from_mapping(value, default_role)
        return [InputMessage(role=role, parts=parts)]
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        messages: list[InputMessage] = []
        for item in value:
            if isinstance(item, Mapping):
                role, parts = _message_from_mapping(item, default_role)
                messages.append(InputMessage(role=role, parts=parts))
            elif isinstance(item, str):
                messages.append(
                    InputMessage(
                        role=default_role,
                        parts=cast(
                            list[MessagePart], [Text(content=item)]
                        ),
                    )
                )
        return messages
    return [
        InputMessage(
            role=default_role,
            parts=cast(
                list[MessagePart], [Text(content=safe_json_dumps(value))]
            ),
        )
    ]


def _output_messages(
    value: Any,
    default_role: str = "assistant",
    finish_reason: str | None = None,
) -> list[OutputMessage]:
    messages = _input_messages(value, default_role=default_role)
    return [
        OutputMessage(
            role=message.role,
            parts=message.parts,
            finish_reason=finish_reason or "stop",
        )
        for message in messages
    ]


def _usage_from_response(response: Any) -> Any:
    if isinstance(response, Mapping):
        return response.get("usage") or response.get("usage_metadata")
    return getattr(response, "usage", None) or getattr(
        response, "usage_metadata", None
    )


def _apply_token_usage(
    invocation: InferenceInvocation,
    usage_data: Any,
) -> None:
    if not isinstance(usage_data, Mapping):
        return
    input_tokens = _first_not_none(
        usage_data.get("prompt_tokens"),
        usage_data.get("prompt_token_count"),
        usage_data.get("input_tokens"),
    )
    output_tokens = _first_not_none(
        usage_data.get("completion_tokens"),
        usage_data.get("candidates_token_count"),
        usage_data.get("output_tokens"),
    )
    cached_tokens = _first_not_none(
        usage_data.get("cached_tokens"),
        usage_data.get("cached_prompt_tokens"),
    )
    if input_tokens is not None:
        invocation.input_tokens = int(input_tokens)
    if output_tokens is not None:
        invocation.output_tokens = int(output_tokens)
    if cached_tokens is not None:
        invocation.cache_read_input_tokens = int(cached_tokens)


class CrewAIInferenceEventListener(BaseEventListener):
    """Convert CrewAI LLM events into GenAI inference spans.

    This ports one span type from the donated OpenInference implementation:
    LLM call events become ``TelemetryHandler.inference()`` invocations. It
    does not wrap or replace CrewAI methods.
    """

    def __init__(self, telemetry_handler: TelemetryHandler) -> None:
        self._telemetry_handler = telemetry_handler
        self._invocations: dict[str, InferenceInvocation] = {}
        self._handlers: list[tuple[type[Any], Any]] = []
        self._event_bus: CrewAIEventsBus | None = None
        super().__init__()

    def setup_listeners(self, crewai_event_bus: CrewAIEventsBus) -> None:
        self._event_bus = crewai_event_bus
        self._register(LLMCallStartedEvent, self._on_llm_started)
        self._register(LLMCallCompletedEvent, self._on_llm_completed)
        self._register(LLMCallFailedEvent, self._on_llm_failed)

    def shutdown(self) -> None:
        if self._event_bus is not None:
            for event_cls, handler in self._handlers:
                try:
                    self._event_bus.off(event_cls, handler)
                except Exception:
                    logger.debug(
                        "Failed to unregister CrewAI event handler",
                        exc_info=True,
                    )
        for invocation in self._invocations.values():
            invocation.fail(
                RuntimeError(
                    "CrewAI instrumentation shut down before LLM call completed"
                )
            )
        self._invocations.clear()
        self._handlers.clear()
        self._event_bus = None

    def _register(self, event_cls: type[Any], handler: Any) -> None:
        if self._event_bus is None:
            raise RuntimeError("CrewAI event bus is not initialized")
        decorated = self._event_bus.on(event_cls)(handler)
        self._handlers.append((event_cls, decorated))

    def _on_llm_started(self, source: Any, event: LLMCallStartedEvent) -> None:
        if _is_suppressed():
            return
        event_id = _event_id(event)
        if event_id is None:
            return

        model = str(getattr(event, "model", None) or "unknown")
        invocation = self._telemetry_handler.inference(
            provider="crewai",
            request_model=model if model != "unknown" else None,
        )
        invocation.input_messages = _input_messages(
            getattr(event, "messages", None)
        )

        tools = getattr(event, "tools", None)
        if tools is not None:
            invocation.attributes["crewai.llm.tools"] = safe_json_dumps(tools)
        available_functions = getattr(event, "available_functions", None)
        if available_functions is not None:
            invocation.attributes["crewai.llm.available_functions"] = (
                safe_json_dumps(available_functions)
            )
        self._invocations[event_id] = invocation

    def _on_llm_completed(
        self,
        source: Any,
        event: LLMCallCompletedEvent,
    ) -> None:
        if _is_suppressed():
            return
        event_id = _started_event_id(event)
        if event_id is None:
            return
        invocation = self._invocations.pop(event_id, None)
        if invocation is None:
            return
        response = getattr(event, "response", None)
        finish_reason = getattr(event, "finish_reason", None)
        invocation.output_messages = _output_messages(
            response, finish_reason=finish_reason
        )
        if finish_reason:
            invocation.finish_reasons = [finish_reason]
        response_id = getattr(event, "response_id", None)
        if response_id:
            invocation.response_id = response_id
        # The event's usage field is authoritative; fall back to a usage
        # payload embedded in the response for legacy emitters.
        _apply_token_usage(
            invocation,
            getattr(event, "usage", None) or _usage_from_response(response),
        )
        invocation.stop()

    def _on_llm_failed(self, source: Any, event: LLMCallFailedEvent) -> None:
        if _is_suppressed():
            return
        event_id = _started_event_id(event)
        if event_id is None:
            return
        invocation = self._invocations.pop(event_id, None)
        if invocation is None:
            return
        error = getattr(event, "error", None) or RuntimeError(
            "CrewAI LLM call failed"
        )
        if isinstance(error, BaseException):
            invocation.fail(error)
        else:
            invocation.fail(RuntimeError(str(error)))
