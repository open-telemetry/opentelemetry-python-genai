# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helper utilities for Anthropic instrumentation."""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
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
)

from opentelemetry.util.genai.types import (
    BlobPart,
    CompactionPart,
    FilePart,
    GenericPart,
    MessagePart,
    ReasoningPart,
    ServerToolCallPart,
    ServerToolCallResponsePart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anthropic.types import (
        ContentBlock,
        ContentBlockParam,
        RawContentBlockDelta,
    )
    from anthropic.types.beta import (
        BetaContentBlock,
        BetaContentBlockParam,
        BetaRedactedThinkingBlock,
        BetaTextBlock,
        BetaThinkingBlock,
        BetaToolUseBlock,
    )
    from anthropic.types.beta import (
        BetaMessage as AnthropicBetaMessage,
    )
else:
    try:
        import anthropic.types.beta as _beta_types
    except (ImportError, AttributeError):
        _beta_types = None

    def _get_beta_type(name: str) -> type:
        cls = getattr(_beta_types, name, None)
        return cls if isinstance(cls, type) else type(name, (), {})

    AnthropicBetaMessage = _get_beta_type("BetaMessage")
    BetaRedactedThinkingBlock = _get_beta_type("BetaRedactedThinkingBlock")
    BetaTextBlock = _get_beta_type("BetaTextBlock")
    BetaThinkingBlock = _get_beta_type("BetaThinkingBlock")
    BetaToolUseBlock = _get_beta_type("BetaToolUseBlock")


__all__ = [
    "AnthropicBetaMessage",
    "convert_content_to_parts",
    "create_stream_block_state",
    "is_anthropic_async_stream",
    "is_anthropic_stream",
    "normalize_finish_reason",
    "stream_block_state_to_part",
    "update_stream_block_state",
]


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


def _decode_base64(data: str) -> bytes | None:
    try:
        return base64.b64decode(data)
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _extract_base64_blob(source: object, modality: str) -> BlobPart | None:
    """Extract a BlobPart from a base64-encoded source dict."""
    if not isinstance(source, dict):
        return None
    # source is a TypedDict (e.g. Base64ImageSourceParam) narrowed to dict;
    # pyright cannot infer value types from isinstance-narrowed dicts.
    data: object = source.get("data")  # type: ignore[reportUnknownMemberType]
    if not isinstance(data, str):
        return None
    decoded = _decode_base64(data)
    if decoded is None:
        return None
    media_type: object = source.get("media_type")  # type: ignore[reportUnknownMemberType]
    return BlobPart(
        mime_type=media_type if isinstance(media_type, str) else None,
        modality=modality,
        content=decoded,
    )


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

    if block_type in ("server_tool_use", "mcp_tool_use"):
        server_tool_call: dict[str, Any] = {
            "type": block_type,
            "arguments": block.get("input"),
        }
        for key in ("caller", "server_name"):
            value = block.get(key)
            if value is not None:
                server_tool_call[key] = value
        return ServerToolCallPart(
            name=str(block.get("name", "")),
            server_tool_call=server_tool_call,
            id=str(block.get("id", "")),
        )

    if block_type == "tool_result":
        return ToolCallResponsePart(
            response=block.get("content"),
            id=str(block.get("tool_use_id", "")),
        )

    if isinstance(block_type, str) and block_type.endswith("_tool_result"):
        return ServerToolCallResponsePart(
            server_tool_call_response={
                key: value
                for key, value in block.items()
                if key != "tool_use_id"
            },
            id=str(block.get("tool_use_id", "")),
        )

    if block_type in ("thinking", "redacted_thinking"):
        thinking = block.get("thinking") or block.get("data")
        return ReasoningPart(
            content=str(thinking) if thinking is not None else ""
        )

    if block_type == "container_upload":
        file_id = block.get("file_id")
        if isinstance(file_id, str):
            return FilePart(
                mime_type=None,
                modality="document",
                file_id=file_id,
            )

    if block_type == "compaction":
        content = block.get("content")
        return CompactionPart(
            content=content if isinstance(content, str) else None
        )

    if block_type in ("image", "audio", "video", "document", "file"):
        part = _extract_base64_blob(block.get("source"), str(block_type))
        if part is not None:
            return part

    return (
        GenericPart(type=str(block_type)) if block_type is not None else None
    )


def _convert_content_block_to_part(
    block: ContentBlock
    | ContentBlockParam
    | BetaContentBlock
    | BetaContentBlockParam,
) -> MessagePart | None:
    """Convert an Anthropic content block to a MessagePart."""
    if isinstance(block, (TextBlock, BetaTextBlock)):
        return TextPart(content=block.text)

    if isinstance(block, (ToolUseBlock, BetaToolUseBlock)):
        return ToolCallRequestPart(
            arguments=block.input, name=block.name, id=block.id
        )

    if isinstance(
        block,
        (
            ThinkingBlock,
            RedactedThinkingBlock,
            BetaThinkingBlock,
            BetaRedactedThinkingBlock,
        ),
    ):
        content = (
            block.thinking
            if isinstance(block, (ThinkingBlock, BetaThinkingBlock))
            else block.data
        )
        return ReasoningPart(content=content)

    model_dump = getattr(block, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return _convert_dict_block_to_part(cast(Mapping[str, Any], dumped))

    if not hasattr(block, "get"):
        return None
    return _convert_dict_block_to_part(cast(Mapping[str, Any], block))


def convert_content_to_parts(
    content: str
    | Iterable[
        ContentBlock
        | ContentBlockParam
        | BetaContentBlock
        | BetaContentBlockParam
    ]
    | None,
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
