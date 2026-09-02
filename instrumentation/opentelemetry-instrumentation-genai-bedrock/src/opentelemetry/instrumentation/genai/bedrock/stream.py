# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from botocore.eventstream import EventStream

from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import SyncStreamWrapper
from opentelemetry.util.genai.types import (
    MessagePart,
    OutputMessage,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallRequestPart,
)

from .extractors import _is_dict, map_finish_reason


class BedrockConverseStreamWrapper(SyncStreamWrapper[dict[str, Any]]):
    """Wrapper for Bedrock converse_stream EventStream."""

    def __init__(
        self,
        stream: EventStream,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_role = Role.ASSISTANT.value
        self._self_stop_reason: str | None = None
        self._self_input_tokens: int | None = None
        self._self_output_tokens: int | None = None
        self._self_cache_read_input_tokens: int | None = None
        self._self_cache_creation_input_tokens: int | None = None
        self._self_text_blocks: dict[int, str] = {}
        self._self_reasoning_blocks: dict[int, str] = {}
        self._self_tool_blocks: dict[int, dict[str, Any]] = {}
        self._self_all_block_indices: list[int] = []

    def _process_chunk(self, chunk: dict[str, Any]) -> None:
        if "messageStart" in chunk and "role" in chunk["messageStart"]:
            self._self_role = chunk["messageStart"]["role"]

        if self._self_capture_content and "contentBlockStart" in chunk:
            cb_start = chunk["contentBlockStart"]
            idx = cb_start.get("contentBlockIndex", 0)
            if idx not in self._self_all_block_indices:
                self._self_all_block_indices.append(idx)
            start = cb_start.get("start", {})
            if "toolUse" in start:
                tool_use = start["toolUse"]
                self._self_tool_blocks[idx] = {
                    "toolUseId": tool_use.get("toolUseId"),
                    "name": tool_use.get("name", ""),
                    "input_chunks": [],
                }

        if self._self_capture_content and "contentBlockDelta" in chunk:
            cb_delta = chunk["contentBlockDelta"]
            idx = cb_delta.get("contentBlockIndex", 0)
            if idx not in self._self_all_block_indices:
                self._self_all_block_indices.append(idx)
            delta = cb_delta.get("delta", {})
            if "text" in delta:
                self._self_text_blocks[idx] = (
                    self._self_text_blocks.get(idx, "") + delta["text"]
                )
            elif "reasoningContent" in delta:
                rc_delta = delta["reasoningContent"]
                if _is_dict(rc_delta) and "text" in rc_delta:
                    self._self_reasoning_blocks[idx] = (
                        self._self_reasoning_blocks.get(idx, "")
                        + rc_delta["text"]
                    )
            elif "toolUse" in delta:
                tool_delta = delta["toolUse"]
                input_str = tool_delta.get("input", "")
                if idx not in self._self_tool_blocks:
                    self._self_tool_blocks[idx] = {
                        "toolUseId": None,
                        "name": "",
                        "input_chunks": [],
                    }
                self._self_tool_blocks[idx]["input_chunks"].append(input_str)

        if "messageStop" in chunk:
            msg_stop = chunk["messageStop"]
            if "stopReason" in msg_stop:
                self._self_stop_reason = msg_stop["stopReason"]

        if "metadata" in chunk:
            metadata = chunk["metadata"]
            usage = metadata.get("usage", {})
            if "inputTokens" in usage:
                self._self_input_tokens = usage["inputTokens"]
            if "outputTokens" in usage:
                self._self_output_tokens = usage["outputTokens"]
            if "cacheReadInputTokens" in usage:
                self._self_cache_read_input_tokens = usage[
                    "cacheReadInputTokens"
                ]
            if "cacheWriteInputTokens" in usage:
                self._self_cache_creation_input_tokens = usage[
                    "cacheWriteInputTokens"
                ]

    def _on_stream_end(self) -> None:
        finish_reason = map_finish_reason(self._self_stop_reason)
        if finish_reason:
            self._self_invocation.finish_reasons = [finish_reason]

        if self._self_capture_content:
            parts: list[MessagePart] = []
            for idx in sorted(self._self_all_block_indices):
                if idx in self._self_text_blocks:
                    parts.append(TextPart(content=self._self_text_blocks[idx]))
                elif idx in self._self_reasoning_blocks:
                    parts.append(
                        ReasoningPart(content=self._self_reasoning_blocks[idx])
                    )
                elif idx in self._self_tool_blocks:
                    tool_info = self._self_tool_blocks[idx]
                    input_chunks = tool_info["input_chunks"]
                    raw_input = "".join(input_chunks)
                    args: object
                    if raw_input:
                        try:
                            args = json.loads(raw_input)
                        except Exception:
                            args = raw_input
                    else:
                        args = {}
                    parts.append(
                        ToolCallRequestPart(
                            id=tool_info["toolUseId"],
                            name=tool_info["name"],
                            arguments=args,
                        )
                    )

            if parts or finish_reason:
                self._self_invocation.output_messages = [
                    OutputMessage(
                        role=self._self_role,
                        parts=parts,
                        finish_reason=finish_reason or "stop",
                    )
                ]
        self._self_invocation.input_tokens = self._self_input_tokens
        self._self_invocation.output_tokens = self._self_output_tokens
        self._self_invocation.cache_read_input_tokens = (
            self._self_cache_read_input_tokens
        )
        self._self_invocation.cache_creation_input_tokens = (
            self._self_cache_creation_input_tokens
        )

        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)
