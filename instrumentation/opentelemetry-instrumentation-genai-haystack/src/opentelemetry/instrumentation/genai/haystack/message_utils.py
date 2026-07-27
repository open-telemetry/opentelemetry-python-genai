# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Maps Haystack ``ChatMessage`` / ``Document`` / ``Tool`` shapes onto the
``opentelemetry.util.genai.types`` message model.

Reference: https://github.com/open-telemetry/semantic-conventions-genai/tree/main/docs/gen-ai
(``gen-ai-input-messages.json`` / ``gen-ai-output-messages.json`` /
``gen-ai-tool-definitions.json``).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

_FINISH_REASON_MAP = {
    "stop": "stop",
    "length": "length",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    "tool_call": "tool_calls",
}


def _normalize_finish_reason(finish_reason: str | None) -> str:
    if finish_reason is None:
        return "stop"
    return _FINISH_REASON_MAP.get(finish_reason, finish_reason)


def _chat_message_parts(message: Any) -> list[MessagePart]:
    """Build the ``parts`` list for a single Haystack ``ChatMessage``."""
    parts: list[MessagePart] = []
    for tool_call in message.tool_calls or []:
        parts.append(
            ToolCallRequest(
                name=tool_call.tool_name,
                arguments=tool_call.arguments,
                id=tool_call.id,
            )
        )
    for tool_call_result in message.tool_call_results or []:
        origin = tool_call_result.origin
        parts.append(
            ToolCallResponse(
                response=tool_call_result.result,
                id=origin.id if origin is not None else None,
            )
        )
    text = message.text
    if text is not None:
        parts.append(Text(content=text))
    return parts


def to_input_message(message: Any) -> InputMessage:
    """Convert a single Haystack ``ChatMessage`` into an ``InputMessage``."""
    return InputMessage(
        role=message.role.value, parts=_chat_message_parts(message)
    )


def to_input_messages(messages: Sequence[Any]) -> list[InputMessage]:
    """Convert a ``List[ChatMessage]`` (a ChatGenerator's ``messages`` argument)."""
    return [to_input_message(message) for message in messages]


def prompt_to_input_messages(prompt: str) -> list[InputMessage]:
    """Convert a plain-text ``Generator`` prompt string into a single ``InputMessage``."""
    return [InputMessage(role="user", parts=[Text(content=prompt)])]


def chat_replies_to_output_messages(
    replies: Sequence[Any],
) -> list[OutputMessage]:
    """Convert a ChatGenerator's ``replies: List[ChatMessage]`` into ``OutputMessage``\\ s."""
    output_messages: list[OutputMessage] = []
    for reply in replies:
        meta = reply.meta if isinstance(reply.meta, dict) else {}
        finish_reason = _normalize_finish_reason(meta.get("finish_reason"))
        output_messages.append(
            OutputMessage(
                role=reply.role.value,
                parts=_chat_message_parts(reply),
                finish_reason=finish_reason,
            )
        )
    return output_messages


def text_replies_to_output_messages(
    replies: Sequence[str],
) -> list[OutputMessage]:
    """Convert a text ``Generator``'s ``replies: List[str]`` into ``OutputMessage``\\ s."""
    return [
        OutputMessage(
            role="assistant", parts=[Text(content=reply)], finish_reason="stop"
        )
        for reply in replies
    ]


def tool_to_definition(tool: Any) -> ToolDefinition:
    """Convert a Haystack ``Tool`` into a ``FunctionToolDefinition``."""
    return FunctionToolDefinition(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
    )


def tools_to_definitions(tools: Any) -> list[ToolDefinition] | None:
    """Best-effort conversion of a ChatGenerator's ``tools`` argument.

    Only handles the typed ``haystack.tools.Tool`` / ``Toolset`` objects a
    caller passes as the dedicated ``tools`` run parameter. Raw provider-
    format tool dicts passed through ``generation_kwargs={"tools": [...]}}``
    are not typed Haystack objects and have no reliable common shape across
    generators.
    """
    if not tools:
        return None
    resolved = getattr(tools, "tools", tools)  # Toolset -> underlying list
    definitions: list[ToolDefinition] = []
    for tool in resolved:
        name = getattr(tool, "name", None)
        if name is None:
            continue
        definitions.append(tool_to_definition(tool))
    return definitions or None


def documents_to_retrieval_documents(
    documents: Sequence[Any],
) -> list[Mapping[str, Any]]:
    """Convert ``List[Document]`` into the ``gen_ai.retrieval.documents`` shape.

    The semconv ``RetrievalDocument`` model requires ``id`` and ``score`` and
    allows extra properties (``ConfigDict(extra="allow")``); ``content`` is
    passed through as one such extra field, matching what Haystack's
    ``Document`` calls it.
    """
    retrieval_documents: list[Mapping[str, Any]] = []
    for document in documents:
        entry: dict[str, Any] = {}
        if document.id is not None:
            entry["id"] = document.id
        if document.score is not None:
            entry["score"] = document.score
        if document.content is not None:
            entry["content"] = document.content
        retrieval_documents.append(entry)
    return retrieval_documents
