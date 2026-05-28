# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Optional, cast

from langchain_core.messages import AIMessage

from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
)


def make_input_message(data: Any) -> list[InputMessage]:
    """Create structured input message with full data as JSON."""
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    input_messages: list[InputMessage] = []
    messages: Any = data_dict.get("messages")
    if messages is not None:
        for msg in messages:
            content: Any = getattr(msg, "content", "")
            if content and isinstance(content, str):
                input_message = InputMessage(
                    role="user", parts=[Text(content)]
                )
                input_messages.append(input_message)
        return input_messages
    # Fallback: serialize non-message state fields as input.
    # Common in LangGraph where nodes use structured state fields
    # (e.g., user_query) rather than a message list.
    exclude_keys = {"messages", "intermediate_steps"}
    input_data: dict[str, Any] = {
        k: v
        for k, v in data_dict.items()
        if k not in exclude_keys and v is not None
    }
    if input_data:
        serialized = serialize(input_data)
        if serialized:
            return [InputMessage(role="user", parts=[Text(serialized)])]
    return input_messages


def make_output_message(data: Any) -> list[OutputMessage]:
    """Create structured output message with full data as JSON."""
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    output_messages: list[OutputMessage] = []
    messages: list[Any] | None = data_dict.get("messages")
    if messages is None:
        return []
    for msg in messages:
        content: Any = getattr(msg, "content", "")
        if content and isinstance(msg, AIMessage) and isinstance(content, str):
            output_message = OutputMessage(
                role="assistant",
                parts=[Text(content)],
                finish_reason="stop",
            )
            output_messages.append(output_message)
    return output_messages


def make_last_output_message(data: Any) -> list[OutputMessage]:
    """Extract only the last AI message as the output.

    For Workflow and AgentInvocation spans, the final AI message best represents
    the actual output. Intermediate AI messages (e.g., tool-call decisions) are
    already captured in child LLM invocation spans.
    """
    all_messages = make_output_message(data)
    if all_messages:
        return [all_messages[-1]]
    return []


def serialize(obj: Any) -> Optional[str]:
    """Serialize object to JSON string.

    Uses default=str to handle non-JSON-serializable objects (like LangChain
    message objects) by converting them to their string representation while
    keeping the overall structure as valid JSON.
    """
    if obj is None:
        return None
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
