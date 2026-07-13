# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Optional, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolMessage,
    convert_to_messages,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

# Mapping from LangChain ``ls_provider`` metadata values to the well-known
# ``gen_ai.provider.name`` values defined by the GenAI semantic conventions.
_PROVIDER_NAME_OVERRIDES: dict[str, str] = {
    "amazon_bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "bedrock_converse": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "azure_openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "vertexai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_vertexai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google_genai": GenAIAttributes.GenAiProviderNameValues.GCP_GEN_AI.value,
    "google_generativeai": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "mistralai": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "mistral": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
}


def normalize_provider(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    """Return the spec ``gen_ai.provider.name`` value derived from metadata.

    Returns ``None`` when no provider can be determined; callers decide how
    to handle that (typically by skipping the span).
    """
    if not metadata:
        return None
    raw = metadata.get("ls_provider")
    if not isinstance(raw, str) or not raw:
        return None
    return _PROVIDER_NAME_OVERRIDES.get(raw, raw)


# LangChain ``BaseMessage.type`` -> spec ``role`` value. Anything not in the
# map is passed through unchanged so future LangChain message types still emit
# telemetry without code changes here.
_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "ai": "assistant",
    "function": "tool",
}


def _normalize_role(message: BaseMessage) -> str:
    return _ROLE_MAP.get(message.type, message.type)


def _content_to_parts(
    content: str | list[str | dict[str, Any]],
) -> list[MessagePart]:
    """Convert a LangChain message ``content`` payload into ``MessagePart`` s.

    Content may be a plain string or a list of provider-specific block dicts
    (e.g. Anthropic structured content). We extract :class:`Text` and
    :class:`Reasoning` parts; ``tool_use`` blocks are intentionally ignored
    here because LangChain consolidates them into ``message.tool_calls`` which
    is read separately.
    """
    parts: list[MessagePart] = []
    if isinstance(content, str):
        if content:
            parts.append(Text(content=content))
        return parts
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(Text(content=item))
            continue
        block_type = item.get("type")
        if block_type == "text":
            text_value = item.get("text")
            if isinstance(text_value, str) and text_value:
                parts.append(Text(content=text_value))
        elif block_type in ("thinking", "reasoning"):
            reasoning_value = (
                item.get("thinking")
                or item.get("reasoning")
                or item.get("text")
            )
            if isinstance(reasoning_value, str) and reasoning_value:
                parts.append(Reasoning(content=reasoning_value))
    return parts


def _ai_message_parts(message: AIMessage) -> list[MessagePart]:
    """Build :class:`MessagePart` s for an :class:`AIMessage`.

    Includes any text/reasoning content followed by a
    :class:`ToolCallRequest` for each entry in ``message.tool_calls``.
    """
    parts: list[MessagePart] = _content_to_parts(message.content)
    for call in message.tool_calls:
        name = call["name"]
        if not name:
            continue
        parts.append(
            ToolCallRequest(
                arguments=call["args"],
                name=name,
                id=call["id"],
            )
        )
    return parts


def _tool_message_parts(message: ToolMessage) -> list[MessagePart]:
    """Build :class:`MessagePart` s for a :class:`ToolMessage` (tool result)."""
    tool_call_id = getattr(message, "tool_call_id", None)
    return [
        ToolCallResponse(
            response=message.content,
            id=tool_call_id if isinstance(tool_call_id, str) else None,
        )
    ]


def _message_parts(message: BaseMessage) -> list[MessagePart]:
    if isinstance(message, ToolMessage):
        return _tool_message_parts(message)
    if isinstance(message, AIMessage):
        return _ai_message_parts(message)
    return _content_to_parts(message.content)


def to_input_messages(
    messages: Iterable[Any],
) -> list[InputMessage]:
    """Convert LangChain messages into spec-conformant ``InputMessage`` s."""
    try:
        normalized_messages: Iterable[BaseMessage] = convert_to_messages(
            list(messages)
        )
    except Exception:  # pylint: disable=broad-except
        normalized_messages = [
            m for m in messages if isinstance(m, BaseMessage)
        ]
    result: list[InputMessage] = []
    for message in normalized_messages:
        parts = _message_parts(message)
        if not parts:
            continue
        result.append(InputMessage(role=_normalize_role(message), parts=parts))
    return result


def to_output_messages(
    messages: Iterable[BaseMessage],
    *,
    finish_reason: str = "",
) -> list[OutputMessage]:
    """Convert LangChain ``AIMessage`` instances into ``OutputMessage`` s.

    Non-``AIMessage`` entries are skipped: only assistant turns are recorded
    as ``gen_ai.output.messages``. Tool execution results belong on the
    *input* side of the next inference call, not the output side of the
    previous one.
    """
    result: list[OutputMessage] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        parts = _ai_message_parts(message)
        if not parts:
            continue
        result.append(
            OutputMessage(
                role=_normalize_role(message),
                parts=parts,
                finish_reason=finish_reason,
            )
        )
    return result


def _get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)

    return getattr(obj, property_name, None)


def prepare_tool_definitions(tools: list[Any]) -> list[ToolDefinition] | None:
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = _get_property_value(tool, "type")
        if tool_type == "function":
            func = _get_property_value(tool, "function")
            if func:
                func_name = _get_property_value(func, "name")
                func_description = _get_property_value(func, "description")
                definitions.append(
                    FunctionToolDefinition(
                        name=str(func_name) if func_name is not None else "",
                        description=str(func_description)
                        if func_description is not None
                        else None,
                        parameters=_get_property_value(func, "parameters"),
                    )
                )
    return definitions or None


def make_input_message(data: Any) -> list[InputMessage]:
    """Build ``InputMessage`` s from a workflow/agent input mapping.

    When ``data['messages']`` is present, every LangChain ``BaseMessage`` in it
    is converted via :func:`to_input_messages` (which preserves the original
    role: a prior ``AIMessage`` becomes ``role='assistant'``, a
    ``SystemMessage`` becomes ``role='system'``, and so on) and includes
    tool-call structure.

    When no ``messages`` key exists (common in LangGraph state dicts), the
    remaining state fields are serialized as JSON and emitted as a single
    user-role :class:`Text` part.
    """
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: Any = data_dict.get("messages")
    if messages is not None:
        if isinstance(messages, (str, bytes)) or not isinstance(
            messages, Iterable
        ):
            return []
        return to_input_messages(cast(Iterable[BaseMessage], messages))
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
    return []


def make_output_message(data: Any) -> list[OutputMessage]:
    """Build ``OutputMessage`` s from a workflow/agent output mapping.

    Only ``AIMessage`` entries become outputs. ``finish_reason`` is left
    empty: the underlying per-LLM-call finish reasons are recorded on child
    inference spans, and util-genai filters empty values out of
    ``gen_ai.response.finish_reasons``.
    """
    if not isinstance(data, dict):
        return []
    data_dict = cast(dict[str, Any], data)
    messages: Any = data_dict.get("messages")
    if (
        messages is None
        or isinstance(messages, (str, bytes))
        or not isinstance(messages, Iterable)
    ):
        return []
    return to_output_messages(cast(Iterable[BaseMessage], messages))


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
