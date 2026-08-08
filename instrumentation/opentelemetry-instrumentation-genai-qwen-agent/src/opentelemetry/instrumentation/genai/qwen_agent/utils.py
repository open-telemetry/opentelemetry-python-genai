# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversion helpers between qwen-agent message types and
``opentelemetry.util.genai`` semantic convention types."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)

if TYPE_CHECKING:
    from qwen_agent.llm.schema import ContentItem, Message

    # qwen-agent accepts a ``Message`` or a plain dict wherever a message or a
    # content item is expected and normalizes them internally, so both shapes
    # reach the instrumented methods.
    QwenMessage = Message | dict[str, Any]
    QwenContentItem = ContentItem | dict[str, Any]

_logger = logging.getLogger(__name__)


def _field_value(value: Any, *names: str) -> Any:
    """Read the first present field from a dict or SDK object."""
    if value is None:
        return None

    for name in names:
        if isinstance(value, dict):
            mapping = cast(dict[str, Any], value)
            if name in mapping:
                return mapping[name]
            continue

        attr_value = getattr(value, name, None)
        if attr_value is not None:
            return attr_value

        get_method = getattr(value, "get", None)
        if callable(get_method):
            try:
                got_value = get_method(name)
            except Exception:  # pylint: disable=broad-exception-caught
                got_value = None
            if got_value is not None:
                return got_value

    return None


def _extract_content_text(content: str | list[QwenContentItem] | None) -> str:
    """Extract text from a qwen-agent Message content field.

    Content can be a plain string or a list of ContentItem.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
        return "\n".join(texts)
    return str(content) if content else ""


def find_tool_call_id(
    messages: list[QwenMessage] | None, tool_name: str
) -> str | None:
    """Best-effort lookup of the current tool call's function id.

    ``FnCallAgent`` passes the message history to ``_call_tool``; the call
    being executed corresponds to the first ``function_call`` message for
    ``tool_name`` that has no function-result message answering its
    ``function_id`` yet.
    """
    if not isinstance(messages, list):
        return None

    responded: set[str] = set()
    pending: list[str] = []
    for msg in messages:
        extra = _field_value(msg, "extra")
        function_id = (
            _field_value(extra, "function_id") if extra is not None else None
        )
        if _field_value(msg, "role") in ("function", "tool"):
            if function_id:
                responded.add(str(function_id))
            continue
        function_call = _field_value(msg, "function_call")
        if (
            function_call
            and _field_value(function_call, "name") == tool_name
            and function_id
        ):
            pending.append(str(function_id))

    for function_id in pending:
        if function_id not in responded:
            return function_id
    return None


def _function_call_part(function_call: Any) -> ToolCallRequest:
    name = _field_value(function_call, "name") or ""
    arguments = _field_value(function_call, "arguments") or "{}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    return ToolCallRequest(name=name, arguments=arguments, id=None)


def _tool_call_response_id(msg: Any) -> str:
    """Extract a tool call id for a function/tool role message."""
    tool_call_id = _field_value(msg, "id") or ""
    if not tool_call_id:
        extra = _field_value(msg, "extra")
        if extra is not None:
            tool_call_id = _field_value(extra, "function_id") or ""
    if not tool_call_id:
        tool_call_id = _field_value(msg, "name") or ""
    return tool_call_id


def convert_to_input_messages(
    messages: QwenMessage | list[QwenMessage] | None,
) -> list[InputMessage]:
    """Convert a qwen-agent Message list to GenAI InputMessage objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    input_messages: list[InputMessage] = []
    for msg in messages:
        try:
            role = _field_value(msg, "role") or "user"
            content = _field_value(msg, "content") or ""
            function_call = _field_value(msg, "function_call")

            parts: list[MessagePart] = []

            if function_call:
                parts.append(_function_call_part(function_call))

            # qwen-agent uses role="function" internally, but the DashScope
            # API converts it to role="tool"; handle both.
            if role in ("function", "tool") and content:
                parts.append(
                    ToolCallResponse(
                        id=_tool_call_response_id(msg),
                        response=_extract_content_text(content),
                    )
                )
            elif content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if parts:
                input_messages.append(InputMessage(role=role, parts=parts))
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Error converting input message", exc_info=True)
            continue

    return input_messages


def convert_to_output_messages(
    messages: QwenMessage | list[QwenMessage] | None,
) -> list[OutputMessage]:
    """Convert qwen-agent response messages to GenAI OutputMessage objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    output_messages: list[OutputMessage] = []
    for msg in messages:
        try:
            content = _field_value(msg, "content") or ""
            function_call = _field_value(msg, "function_call")

            parts: list[MessagePart] = []
            finish_reason = "stop"

            if function_call:
                parts.append(_function_call_part(function_call))
                finish_reason = "tool_calls"

            if content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=finish_reason,
                )
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Error converting output message", exc_info=True)
            continue

    return output_messages


def convert_to_final_output_messages(
    messages: QwenMessage | list[QwenMessage] | None,
) -> list[OutputMessage]:
    """Convert only the final qwen-agent answer to OutputMessage objects.

    An agent run yields the full response history including intermediate
    tool calls and tool results; the agent's output is the last assistant
    text message.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    for msg in reversed(messages):
        try:
            role = _field_value(msg, "role") or "assistant"
            function_call = _field_value(msg, "function_call")
            content = _field_value(msg, "content") or ""

            if role in ("function", "tool") or function_call:
                continue

            if _extract_content_text(content):
                return convert_to_output_messages([msg])
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Error extracting final agent output message", exc_info=True
            )
            continue

    return []


def create_agent_invocation(
    handler: TelemetryHandler,
    agent_instance: Any,
    messages: QwenMessage | list[QwenMessage] | None,
) -> AgentInvocation:
    """Create and start an AgentInvocation for Agent.run()."""
    llm_instance = getattr(agent_instance, "llm", None)
    agent_name = (
        getattr(agent_instance, "name", None) or type(agent_instance).__name__
    )

    # invoke_local_agent() intentionally takes no provider: the INTERNAL
    # invoke_agent span describes in-process agent logic, not a provider call.
    invocation = handler.invoke_local_agent(
        request_model=getattr(llm_instance, "model", None)
        if llm_instance is not None
        else None,
        agent_name=agent_name,
    )
    invocation.agent_description = (
        getattr(agent_instance, "description", None) or None
    )

    if handler.should_capture_content():
        invocation.input_messages = convert_to_input_messages(messages)
        # Agent.system_message is the configured system prompt that
        # qwen-agent prepends to the LLM messages on every run.
        system_message = getattr(agent_instance, "system_message", None)
        if system_message:
            invocation.system_instruction = [Text(content=system_message)]

    return invocation
