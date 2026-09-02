# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeGuard
from urllib.parse import urlparse

from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericPart,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
)


def _is_dict(val: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[Any]]:
    return isinstance(val, list)


_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
}

_DOC_MIME_TYPES: dict[str, str] = {
    "pdf": "application/pdf",
    "csv": "text/csv",
    "doc": "application/msword",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "xls": "application/vnd.ms-excel",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "html": "text/html",
    "txt": "text/plain",
    "md": "text/markdown",
}


def map_finish_reason(stop_reason: str | None) -> str | None:
    """Map Bedrock stopReason to GenAI semantic convention finish_reason."""
    if stop_reason is None:
        return None
    return _FINISH_REASON_MAP.get(stop_reason, stop_reason.lower())


def extract_server_address_and_port(
    endpoint_url: str | None,
) -> tuple[str | None, int | None]:
    """Parse server address and port from a client endpoint URL."""
    if not endpoint_url:
        return None, None
    parsed = urlparse(endpoint_url)
    server_address = parsed.hostname
    server_port = parsed.port
    if server_port is None and parsed.scheme in ("http", "https"):
        server_port = 443 if parsed.scheme == "https" else 80
    return server_address, server_port


def extract_content_block(block: dict[str, Any]) -> MessagePart | None:
    """Map a single Bedrock content block to an OpenTelemetry MessagePart."""
    if "text" in block:
        return TextPart(content=block["text"])

    reasoning = block.get("reasoningContent")
    if _is_dict(reasoning):
        reasoning_text = reasoning.get("reasoningText")
        if _is_dict(reasoning_text) and "text" in reasoning_text:
            return ReasoningPart(content=reasoning_text["text"])
        if "redactedContent" in reasoning:
            return ReasoningPart(content="")

    image = block.get("image")
    if _is_dict(image):
        fmt = image.get("format", "jpeg")
        source = image.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        return BlobPart(
            content=content_bytes,
            mime_type=f"image/{fmt}",
            modality="image",
        )

    document = block.get("document")
    if _is_dict(document):
        fmt = document.get("format", "pdf")
        source = document.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        mime_type = _DOC_MIME_TYPES.get(fmt, f"application/{fmt}")
        return BlobPart(
            content=content_bytes,
            mime_type=mime_type,
            modality="document",
        )

    tool_use = block.get("toolUse")
    if _is_dict(tool_use):
        return ToolCallRequestPart(
            id=tool_use.get("toolUseId"),
            name=tool_use.get("name", ""),
            arguments=tool_use.get("input"),
        )

    tool_result = block.get("toolResult")
    if _is_dict(tool_result):
        return ToolCallResponsePart(
            id=tool_result.get("toolUseId"),
            response=tool_result.get("content"),
        )

    for key in (
        "video",
        "audio",
        "guardContent",
        "cachePoint",
        "citationsContent",
        "searchResult",
        "toolAddition",
        "toolRemoval",
    ):
        if key in block:
            return GenericPart(type=key, value=None)

    return None


def extract_converse_request(
    kwargs: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate request attributes from converse kwargs onto the invocation."""
    inf_config = kwargs.get("inferenceConfig")
    if _is_dict(inf_config):
        invocation.temperature = inf_config.get("temperature")
        invocation.top_p = inf_config.get("topP")
        invocation.max_tokens = inf_config.get("maxTokens")
        invocation.stop_sequences = inf_config.get("stopSequences")
        invocation.top_k = inf_config.get("topK") or inf_config.get("top_k")
        invocation.seed = inf_config.get("seed")

    add_fields = kwargs.get("additionalModelRequestFields")
    if _is_dict(add_fields):
        add_inf = add_fields.get("inferenceConfig")
        top_k = (
            add_fields.get("topK")
            or add_fields.get("top_k")
            or (
                (add_inf.get("topK") or add_inf.get("top_k"))
                if _is_dict(add_inf)
                else None
            )
        )
        if top_k is not None:
            invocation.top_k = top_k
        if "seed" in add_fields:
            invocation.seed = add_fields["seed"]

    # system instruction
    raw_system = kwargs.get("system")
    if capture_content and _is_list(raw_system):
        system_parts: list[MessagePart] = []
        for item in raw_system:
            if _is_dict(item):
                part = extract_content_block(item)
                if part is not None:
                    system_parts.append(part)
        if system_parts:
            invocation.system_instruction = system_parts

    # input messages
    raw_messages = kwargs.get("messages")
    if capture_content and _is_list(raw_messages):
        input_messages: list[InputMessage] = []
        for msg in raw_messages:
            if not _is_dict(msg):
                continue
            role = msg.get("role", Role.USER.value)
            parts: list[MessagePart] = []
            content = msg.get("content")
            if _is_list(content):
                for block in content:
                    if _is_dict(block):
                        part = extract_content_block(block)
                        if part is not None:
                            parts.append(part)
            input_messages.append(InputMessage(role=role, parts=parts))
        invocation.input_messages = input_messages

    # tool definitions
    tool_config = kwargs.get("toolConfig")
    if _is_dict(tool_config):
        tools = tool_config.get("tools")
        if _is_list(tools):
            tool_defs: list[ToolDefinition] = []
            for tool in tools:
                if not _is_dict(tool):
                    continue
                tool_spec = tool.get("toolSpec")
                if _is_dict(tool_spec):
                    name = tool_spec.get("name", "")
                    description = tool_spec.get("description")
                    raw_schema = tool_spec.get("inputSchema", {})
                    params = (
                        raw_schema.get("json", raw_schema)
                        if _is_dict(raw_schema)
                        else raw_schema
                    )
                    tool_defs.append(
                        FunctionToolDefinition(
                            name=name,
                            description=description,
                            parameters=params,
                        )
                    )
                elif "name" in tool and "type" in tool:
                    tool_defs.append(
                        GenericToolDefinition(
                            name=tool["name"],
                            type=tool["type"],
                        )
                    )
            invocation.tool_definitions = tool_defs


def extract_converse_response(
    response: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    stop_reason = response.get("stopReason")
    finish_reason = map_finish_reason(stop_reason)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]

    output = response.get("output")
    if capture_content and _is_dict(output):
        msg = output.get("message")
        if _is_dict(msg):
            role = msg.get("role", Role.ASSISTANT.value)
            parts: list[MessagePart] = []
            content = msg.get("content")
            if _is_list(content):
                for block in content:
                    if _is_dict(block):
                        part = extract_content_block(block)
                        if part is not None:
                            parts.append(part)
            invocation.output_messages = [
                OutputMessage(
                    role=role,
                    parts=parts,
                    finish_reason=finish_reason or "stop",
                )
            ]

    usage = response.get("usage")
    if _is_dict(usage):
        invocation.input_tokens = usage.get("inputTokens")
        invocation.output_tokens = usage.get("outputTokens")
        invocation.cache_read_input_tokens = usage.get("cacheReadInputTokens")
        invocation.cache_creation_input_tokens = usage.get(
            "cacheWriteInputTokens"
        )
