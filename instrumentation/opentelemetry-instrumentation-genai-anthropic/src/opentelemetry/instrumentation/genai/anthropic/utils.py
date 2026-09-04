# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helper utilities for Anthropic instrumentation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, cast

from anthropic.types import (
    InputJSONDelta,
    RedactedThinkingBlock,
    ServerToolUseBlock,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    ToolUseBlock,
    WebSearchToolResultBlock,
)

from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    GenericPart,
    MessagePart,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    UriPart,
)
from opentelemetry.util.genai.utils import decode_base64, image_from_url

if TYPE_CHECKING:
    from anthropic.types import (
        ContentBlock,
        ContentBlockParam,
        RawContentBlockDelta,
    )


def is_anthropic_stream(value: object) -> bool:
    """Whether ``value`` is a sync SDK ``Stream`` we can drive.

    Matched on shape rather than on ``anthropic._streaming.Stream``: its
    metaclass answers ``isinstance`` ``True`` only for the exact class, so the
    check would have to be duck-typed for subclasses anyway, and importing a
    private SDK module would make a rename break instrumentation at import time.
    """
    return (
        hasattr(value, "__next__")
        and callable(getattr(value, "close", None))
        and hasattr(value, "response")
    )


def is_anthropic_async_stream(value: object) -> bool:
    """Whether ``value`` is an async SDK ``AsyncStream`` we can drive.

    See ``is_anthropic_stream`` for why this is matched on shape.
    """
    return (
        hasattr(value, "__anext__")
        and callable(getattr(value, "close", None))
        and hasattr(value, "response")
    )


@dataclass
class StreamBlockState:
    type: str
    text: str = ""
    tool_id: str | None = None
    tool_name: str = ""
    tool_input: dict[str, object] | None = None
    input_json: str = ""
    thinking: str = ""


def normalize_finish_reason(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    normalized = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_call",
    }.get(stop_reason)
    return normalized or stop_reason


def _extract_base64_blob(source: object, modality: str) -> MessagePart | None:
    """Extract a BlobPart from a base64-encoded source dict."""
    if not isinstance(source, dict):
        return None
    source_dict = cast(dict[str, object], source)
    data = source_dict.get("data")
    if not isinstance(data, str):
        if isinstance(data, PathLike) or callable(getattr(data, "read", None)):
            media_type = source_dict.get("media_type")
            return GenericPart(
                type=modality,
                value={
                    "source_type": "base64_file",
                    "mime_type": media_type
                    if isinstance(media_type, str)
                    else None,
                    "input_type": "path"
                    if isinstance(data, PathLike)
                    else "stream",
                },
            )
        return None
    decoded = decode_base64(data)
    if decoded is None:
        return None
    media_type = source_dict.get("media_type")
    return BlobPart(
        mime_type=media_type if isinstance(media_type, str) else None,
        modality=modality,
        content=decoded,
    )


def _extract_image_source(source: object) -> MessagePart | None:
    """Convert an Anthropic image source into a GenAI message part."""
    if not isinstance(source, dict):
        return None
    source_dict = cast(dict[str, object], source)
    source_type = source_dict.get("type")
    if source_type == "base64":
        return _extract_base64_blob(source_dict, "image")
    if source_type == "url":
        url = source_dict.get("url")
        if isinstance(url, str) and url:
            return image_from_url(url)
    if source_type == "file":
        return _extract_file_source(source_dict, "image")
    return None


def _extract_file_source(
    source: Mapping[str, object], modality: str
) -> FilePart | None:
    file_id = source.get("file_id")
    if not isinstance(file_id, str) or not file_id:
        return None
    return FilePart(mime_type=None, modality=modality, file_id=file_id)


def _extract_document_source(source: object) -> list[MessagePart]:
    """Convert an Anthropic document source into GenAI message parts."""
    if not isinstance(source, dict):
        return []
    source_dict = cast(dict[str, object], source)
    source_type = source_dict.get("type")
    if source_type == "base64":
        part = _extract_base64_blob(source_dict, "document")
        return [part] if part is not None else []
    if source_type == "url":
        url = source_dict.get("url")
        if isinstance(url, str) and url:
            return [
                UriPart(
                    mime_type="application/pdf",
                    modality="document",
                    uri=url,
                )
            ]
        return []
    if source_type == "text":
        data = source_dict.get("data")
        if isinstance(data, str):
            return [
                BlobPart(
                    mime_type="text/plain",
                    modality="document",
                    content=data.encode(),
                )
            ]
        return []
    if source_type == "content":
        content = source_dict.get("content")
        if isinstance(content, str):
            return [TextPart(content=content)]
        if isinstance(content, Iterator):
            return []
        if isinstance(content, Iterable):
            return convert_content_to_parts(
                cast("Iterable[ContentBlock | ContentBlockParam]", content)
            )
    if source_type == "file":
        part = _extract_file_source(source_dict, "document")
        return [part] if part is not None else []
    return []


def _convert_document_block(block: Mapping[str, Any]) -> MessagePart | None:
    parts = _extract_document_source(block.get("source"))
    metadata = {
        key: block[key]
        for key in ("title", "context", "citations")
        if block.get(key) is not None
    }
    source = block.get("source")
    source_mapping = (
        cast(Mapping[str, object], source)
        if isinstance(source, Mapping)
        else None
    )
    is_nested = (
        source_mapping is not None and source_mapping.get("type") == "content"
    )
    if metadata or (is_nested and parts):
        return GenericPart(
            type="document",
            value={
                "parts": [asdict(part) for part in parts],
                **metadata,
            },
        )
    return parts[0] if parts else None


def _convert_dict_block_to_part(
    block: Mapping[str, Any],
) -> MessagePart | None:
    """Convert a request-param content block (TypedDict/dict) to a MessagePart."""
    block_type = block.get("type")

    if block_type == "text":
        text = block.get("text")
        return TextPart(content=str(text) if text is not None else "")

    if block_type == "tool_use":
        inp = block.get("input")
        return ToolCallRequestPart(
            arguments=inp if isinstance(inp, dict) else None,
            name=str(block.get("name", "")),
            id=str(block.get("id", "")),
        )

    if block_type == "tool_result":
        return ToolCallResponsePart(
            response=block.get("content"),
            id=str(block.get("tool_use_id", "")),
        )

    if block_type in ("thinking", "redacted_thinking"):
        thinking = block.get("thinking") or block.get("data")
        return ReasoningPart(
            content=str(thinking) if thinking is not None else ""
        )

    if block_type == "image":
        return _extract_image_source(block.get("source"))

    if block_type == "document":
        return _convert_document_block(block)

    if block_type in ("audio", "video", "file"):
        return _extract_base64_blob(block.get("source"), str(block_type))

    return None


def _convert_content_block_to_part(
    block: ContentBlock | ContentBlockParam,
) -> MessagePart | None:
    """Convert an Anthropic content block to a MessagePart."""
    if isinstance(block, TextBlock):
        return TextPart(content=block.text)

    if isinstance(block, (ToolUseBlock, ServerToolUseBlock)):
        return ToolCallRequestPart(
            arguments=block.input, name=block.name, id=block.id
        )

    if isinstance(block, (ThinkingBlock, RedactedThinkingBlock)):
        content = (
            block.thinking if isinstance(block, ThinkingBlock) else block.data
        )
        return ReasoningPart(content=content)

    if isinstance(block, WebSearchToolResultBlock):
        return ToolCallResponsePart(
            response=block.model_dump().get("content"),
            id=block.tool_use_id,
        )

    if not hasattr(block, "get"):
        return None
    return _convert_dict_block_to_part(cast(Mapping[str, Any], block))


def convert_content_to_parts(
    content: str | Iterable[ContentBlock | ContentBlockParam] | None,
) -> list[MessagePart]:
    if content is None:
        return []
    if isinstance(content, str):
        return [TextPart(content=content)]
    parts: list[MessagePart] = []
    for item in content:
        part = _convert_content_block_to_part(item)
        if part is not None:
            parts.append(part)
    return parts


def create_stream_block_state(content_block: ContentBlock) -> StreamBlockState:
    if isinstance(content_block, TextBlock):
        return StreamBlockState(type="text", text=content_block.text)

    if isinstance(content_block, (ToolUseBlock, ServerToolUseBlock)):
        return StreamBlockState(
            type="tool_use",
            tool_id=content_block.id,
            tool_name=content_block.name,
            tool_input=content_block.input,
        )

    if isinstance(content_block, ThinkingBlock):
        return StreamBlockState(
            type="thinking", thinking=content_block.thinking
        )

    if isinstance(content_block, RedactedThinkingBlock):
        return StreamBlockState(type="redacted_thinking")

    return StreamBlockState(type=content_block.type)


def update_stream_block_state(
    state: StreamBlockState, delta: RawContentBlockDelta
) -> None:
    if isinstance(delta, TextDelta):
        state.type = "text"
        state.text += delta.text
    elif isinstance(delta, InputJSONDelta):
        state.type = "tool_use"
        state.input_json += delta.partial_json
    elif isinstance(delta, ThinkingDelta):
        state.type = "thinking"
        state.thinking += delta.thinking


def stream_block_state_to_part(state: StreamBlockState) -> MessagePart | None:
    if state.type == "text":
        return TextPart(content=state.text)

    if state.type == "tool_use":
        arguments: str | dict[str, object] | None = state.tool_input
        if state.input_json:
            try:
                arguments = json.loads(state.input_json)
            except ValueError:
                arguments = state.input_json
        return ToolCallRequestPart(
            arguments=arguments,
            name=state.tool_name,
            id=state.tool_id,
        )

    if state.type in ("thinking", "redacted_thinking"):
        return ReasoningPart(content=state.thinking)

    return None
