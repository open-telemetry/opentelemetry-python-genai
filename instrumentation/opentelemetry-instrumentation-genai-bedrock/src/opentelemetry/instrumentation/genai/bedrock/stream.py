# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
    RemoteAgentInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    MessagePart,
    OutputMessage,
    ReasoningPart,
    Role,
    TextPart,
    ToolCallRequestPart,
)

from .extractors import (
    _first_not_none,
    _is_dict,
    _is_list,
    _parse_body,
    _safe_int,
    extract_embedding_response,
    extract_invoke_model_response,
    map_finish_reason,
)

if TYPE_CHECKING:

    class _ObjectProxy:
        __wrapped__: Any

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy


def _build_stream_parts(
    all_indices: list[int],
    text_blocks: dict[int, str],
    reasoning_blocks: dict[int, str],
    tool_blocks: dict[int, dict[str, Any]],
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for idx in sorted(all_indices):
        if idx in reasoning_blocks:
            parts.append(ReasoningPart(content=reasoning_blocks[idx]))
        if idx in text_blocks:
            parts.append(TextPart(content=text_blocks[idx]))
        if idx in tool_blocks:
            tool_info = tool_blocks[idx]
            raw_input = "".join(tool_info.get("input_chunks", []))
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
                    id=tool_info.get("id") or tool_info.get("toolUseId"),
                    name=tool_info.get("name", ""),
                    arguments=args,
                )
            )
    return parts


class _BedrockConverseStreamMixin:
    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_role: str
    _self_stop_reason: str | None
    _self_input_tokens: int | None
    _self_output_tokens: int | None
    _self_cache_read_input_tokens: int | None
    _self_cache_creation_input_tokens: int | None
    _self_text_blocks: dict[int, str]
    _self_reasoning_blocks: dict[int, str]
    _self_tool_blocks: dict[int, dict[str, Any]]
    _self_all_block_indices: list[int]

    def _init_converse_stream(
        self,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_role = Role.ASSISTANT.value
        self._self_stop_reason = None
        self._self_input_tokens = None
        self._self_output_tokens = None
        self._self_cache_read_input_tokens = None
        self._self_cache_creation_input_tokens = None
        self._self_text_blocks = {}
        self._self_reasoning_blocks = {}
        self._self_tool_blocks = {}
        self._self_all_block_indices = []

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
                self._self_input_tokens = _safe_int(usage["inputTokens"])
            if "outputTokens" in usage:
                self._self_output_tokens = _safe_int(usage["outputTokens"])
            if "cacheReadInputTokens" in usage:
                self._self_cache_read_input_tokens = _safe_int(
                    usage["cacheReadInputTokens"]
                )
            if "cacheWriteInputTokens" in usage:
                self._self_cache_creation_input_tokens = _safe_int(
                    usage["cacheWriteInputTokens"]
                )

    def _on_stream_end(self) -> None:
        finish_reason = map_finish_reason(self._self_stop_reason)
        if finish_reason:
            self._self_invocation.finish_reasons = [finish_reason]

        if self._self_capture_content:
            parts = _build_stream_parts(
                self._self_all_block_indices,
                self._self_text_blocks,
                self._self_reasoning_blocks,
                self._self_tool_blocks,
            )

            if parts or finish_reason:
                self._self_invocation.output_messages = [
                    OutputMessage(
                        role=self._self_role,
                        parts=parts,
                        finish_reason=finish_reason or "error",
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


class BedrockConverseStreamWrapper(
    _BedrockConverseStreamMixin,
    SyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for Bedrock converse_stream EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_converse_stream(invocation, capture_content=capture_content)


class AsyncBedrockConverseStreamWrapper(
    _BedrockConverseStreamMixin,
    AsyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for async Bedrock converse_stream EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_converse_stream(invocation, capture_content=capture_content)


class _BedrockInvokeModelStreamMixin:
    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_role: str
    _self_stop_reason: str | None
    _self_input_tokens: int | None
    _self_output_tokens: int | None
    _self_cache_read_input_tokens: int | None
    _self_cache_creation_input_tokens: int | None
    _self_accumulated_text: list[str]
    _self_text_blocks: dict[int, str]
    _self_reasoning_blocks: dict[int, str]
    _self_tool_blocks: dict[int, dict[str, Any]]
    _self_all_block_indices: list[int]

    def _init_invoke_model_stream(
        self,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_role = "assistant"
        self._self_stop_reason = None
        self._self_input_tokens = None
        self._self_output_tokens = None
        self._self_cache_read_input_tokens = None
        self._self_cache_creation_input_tokens = None
        self._self_accumulated_text = []
        self._self_text_blocks = {}
        self._self_reasoning_blocks = {}
        self._self_tool_blocks = {}
        self._self_all_block_indices = []

    def _process_chunk(self, chunk: dict[str, Any]) -> None:
        raw_bytes = (
            chunk.get("chunk", {}).get("bytes")
            if _is_dict(chunk.get("chunk"))
            else chunk.get("bytes")
        )
        if not raw_bytes:
            return

        chunk_data = _parse_body(raw_bytes)
        if not _is_dict(chunk_data):
            return

        # 1. Check amazon-bedrock-invocationMetrics
        metrics = chunk_data.get("amazon-bedrock-invocationMetrics")
        if _is_dict(metrics):
            if "inputTokenCount" in metrics:
                self._self_input_tokens = _safe_int(metrics["inputTokenCount"])
            if "outputTokenCount" in metrics:
                self._self_output_tokens = _safe_int(
                    metrics["outputTokenCount"]
                )

        # 2. Anthropic Messages format
        msg_type = chunk_data.get("type")
        if msg_type == "message_start":
            message = chunk_data.get("message", {})
            if _is_dict(message):
                if "role" in message:
                    self._self_role = message["role"]
                usage = message.get("usage", {})
                if _is_dict(usage):
                    if "input_tokens" in usage or "inputTokens" in usage:
                        self._self_input_tokens = _safe_int(
                            _first_not_none(
                                usage.get("input_tokens"),
                                usage.get("inputTokens"),
                            )
                        )
                    if (
                        "cache_read_input_tokens" in usage
                        or "cacheReadInputTokens" in usage
                    ):
                        self._self_cache_read_input_tokens = _safe_int(
                            _first_not_none(
                                usage.get("cache_read_input_tokens"),
                                usage.get("cacheReadInputTokens"),
                            )
                        )
                    if (
                        "cache_creation_input_tokens" in usage
                        or "cacheWriteInputTokens" in usage
                    ):
                        self._self_cache_creation_input_tokens = _safe_int(
                            _first_not_none(
                                usage.get("cache_creation_input_tokens"),
                                usage.get("cacheWriteInputTokens"),
                            )
                        )
        elif msg_type == "content_block_start":
            if self._self_capture_content:
                idx = chunk_data.get("index", 0)
                if idx not in self._self_all_block_indices:
                    self._self_all_block_indices.append(idx)
                cb = chunk_data.get("content_block", {})
                if _is_dict(cb):
                    cb_type = cb.get("type")
                    if cb_type == "tool_use":
                        self._self_tool_blocks[idx] = {
                            "id": cb.get("id"),
                            "name": cb.get("name", ""),
                            "input_chunks": [],
                        }
                    elif cb_type == "text" and "text" in cb:
                        self._self_text_blocks[idx] = cb["text"]
                    elif cb_type in ("thinking", "redacted_thinking"):
                        thinking = cb.get("thinking") or cb.get("data") or ""
                        self._self_reasoning_blocks[idx] = thinking
        elif msg_type == "content_block_delta":
            delta = chunk_data.get("delta", {})
            if _is_dict(delta) and self._self_capture_content:
                idx = chunk_data.get("index", 0)
                if idx not in self._self_all_block_indices:
                    self._self_all_block_indices.append(idx)
                delta_type = delta.get("type")
                if delta_type == "text_delta" and "text" in delta:
                    self._self_text_blocks[idx] = (
                        self._self_text_blocks.get(idx, "") + delta["text"]
                    )
                elif delta_type == "thinking_delta" and "thinking" in delta:
                    self._self_reasoning_blocks[idx] = (
                        self._self_reasoning_blocks.get(idx, "")
                        + delta["thinking"]
                    )
                elif (
                    delta_type == "input_json_delta"
                    and "partial_json" in delta
                ):
                    if idx not in self._self_tool_blocks:
                        self._self_tool_blocks[idx] = {
                            "id": None,
                            "name": "",
                            "input_chunks": [],
                        }
                    self._self_tool_blocks[idx]["input_chunks"].append(
                        delta["partial_json"]
                    )
                elif "text" in delta:
                    self._self_text_blocks[idx] = (
                        self._self_text_blocks.get(idx, "") + delta["text"]
                    )
        elif msg_type == "message_delta":
            delta = chunk_data.get("delta", {})
            if _is_dict(delta) and "stop_reason" in delta:
                self._self_stop_reason = delta["stop_reason"]
            usage = chunk_data.get("usage", {})
            if _is_dict(usage) and "output_tokens" in usage:
                self._self_output_tokens = _safe_int(usage["output_tokens"])

        # 3. Legacy Claude / Titan / Llama / Mistral / Cohere stream chunks
        if "completion" in chunk_data and isinstance(
            chunk_data["completion"], str
        ):
            if self._self_capture_content:
                self._self_accumulated_text.append(chunk_data["completion"])
            if "stop_reason" in chunk_data:
                self._self_stop_reason = chunk_data["stop_reason"]
        elif "outputText" in chunk_data and isinstance(
            chunk_data["outputText"], str
        ):
            if self._self_capture_content:
                self._self_accumulated_text.append(chunk_data["outputText"])
            if "completionReason" in chunk_data:
                self._self_stop_reason = chunk_data["completionReason"]
        elif "generation" in chunk_data and isinstance(
            chunk_data["generation"], str
        ):
            if self._self_capture_content:
                self._self_accumulated_text.append(chunk_data["generation"])
            if "stop_reason" in chunk_data:
                self._self_stop_reason = chunk_data["stop_reason"]
        elif (
            "outputs" in chunk_data
            and _is_list(chunk_data["outputs"])
            and chunk_data["outputs"]
        ):
            out = chunk_data["outputs"][0]
            if _is_dict(out):
                if (
                    self._self_capture_content
                    and "text" in out
                    and isinstance(out["text"], str)
                ):
                    self._self_accumulated_text.append(out["text"])
                if "stop_reason" in out and isinstance(
                    out["stop_reason"], str
                ):
                    self._self_stop_reason = out["stop_reason"]
        elif (
            "text" in chunk_data
            and isinstance(chunk_data["text"], str)
            and msg_type is None
        ):
            if self._self_capture_content:
                self._self_accumulated_text.append(chunk_data["text"])
            if chunk_data.get("is_finished"):
                self._self_stop_reason = "COMPLETE"

    def _on_stream_end(self) -> None:
        finish_reason = map_finish_reason(self._self_stop_reason)
        if finish_reason:
            self._self_invocation.finish_reasons = [finish_reason]

        if self._self_input_tokens is not None:
            self._self_invocation.input_tokens = self._self_input_tokens
        if self._self_output_tokens is not None:
            self._self_invocation.output_tokens = self._self_output_tokens
        if self._self_cache_read_input_tokens is not None:
            self._self_invocation.cache_read_input_tokens = (
                self._self_cache_read_input_tokens
            )
        if self._self_cache_creation_input_tokens is not None:
            self._self_invocation.cache_creation_input_tokens = (
                self._self_cache_creation_input_tokens
            )

        if self._self_capture_content:
            parts: list[MessagePart] = []
            if self._self_all_block_indices:
                parts = _build_stream_parts(
                    self._self_all_block_indices,
                    self._self_text_blocks,
                    self._self_reasoning_blocks,
                    self._self_tool_blocks,
                )
            elif self._self_accumulated_text:
                parts.append(
                    TextPart(content="".join(self._self_accumulated_text))
                )

            if parts or finish_reason:
                self._self_invocation.output_messages = [
                    OutputMessage(
                        role=self._self_role,
                        parts=parts,
                        finish_reason=finish_reason or "error",
                    )
                ]

        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class BedrockInvokeModelStreamWrapper(
    _BedrockInvokeModelStreamMixin,
    SyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for Bedrock invoke_model_with_response_stream EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_invoke_model_stream(
            invocation, capture_content=capture_content
        )


class AsyncBedrockInvokeModelStreamWrapper(
    _BedrockInvokeModelStreamMixin,
    AsyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for async Bedrock invoke_model_with_response_stream EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_invoke_model_stream(
            invocation, capture_content=capture_content
        )


class AsyncBedrockStreamingBodyWrapper(_ObjectProxy):
    """Wrapper for aiobotocore's AioStreamingBody that handles telemetry."""

    _self_invocation: InferenceInvocation | EmbeddingInvocation
    _self_response: dict[str, Any]
    _self_capture_content: bool
    _self_chunks: list[bytes]
    _self_finalized: bool

    def __init__(
        self,
        body: Any,
        invocation: InferenceInvocation | EmbeddingInvocation,
        response: dict[str, Any],
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(body)
        self._self_invocation = invocation
        self._self_response = response
        self._self_capture_content = capture_content
        self._self_chunks = []
        self._self_finalized = False

    def _finalize(self, full_bytes: bytes) -> None:
        if self._self_finalized:
            return
        self._self_finalized = True
        self._self_chunks.clear()
        if isinstance(self._self_invocation, EmbeddingInvocation):
            extract_embedding_response(
                self._self_response,
                full_bytes,
                self._self_invocation,
            )
        else:
            extract_invoke_model_response(
                self._self_response,
                full_bytes,
                self._self_invocation,
                capture_content=self._self_capture_content,
            )
        self._self_invocation.stop()

    def _finalize_error(self, exc: BaseException) -> None:
        if self._self_finalized:
            return
        self._self_finalized = True
        self._self_chunks.clear()
        self._self_invocation.fail(exc)

    async def read(self, amt: int | None = None) -> bytes:
        try:
            chunk: bytes = await self.__wrapped__.read(amt)
        except BaseException as exc:
            self._finalize_error(exc)
            raise

        if amt is None:
            self._self_chunks.append(chunk)
            self._finalize(b"".join(self._self_chunks))
        elif not chunk:
            self._finalize(b"".join(self._self_chunks))
        else:
            self._self_chunks.append(chunk)

        return chunk

    async def aclose(self) -> None:
        try:
            if hasattr(self.__wrapped__, "aclose"):
                await self.__wrapped__.aclose()
            elif hasattr(self.__wrapped__, "close"):
                self.__wrapped__.close()
        finally:
            if not self._self_finalized:
                self._finalize(b"".join(self._self_chunks))

    def close(self) -> None:
        try:
            if hasattr(self.__wrapped__, "close"):
                self.__wrapped__.close()
        finally:
            if not self._self_finalized:
                self._finalize(b"".join(self._self_chunks))

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        try:
            if hasattr(self.__wrapped__, "__aexit__"):
                await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
            elif hasattr(self.__wrapped__, "aclose"):
                await self.__wrapped__.aclose()
            elif hasattr(self.__wrapped__, "close"):
                self.__wrapped__.close()
        finally:
            if exc_val is not None:
                self._finalize_error(exc_val)
            elif not self._self_finalized:
                self._finalize(b"".join(self._self_chunks))

    def __del__(self) -> None:
        if not getattr(self, "_self_finalized", True):
            self._self_finalized = True
            self._self_invocation.stop()


class _BedrockAgentEventStreamMixin:
    _self_invocation: RemoteAgentInvocation
    _self_capture_content: bool
    _self_accumulated_text: list[str]

    def _init_agent_stream(
        self,
        invocation: RemoteAgentInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_accumulated_text = []

    def _process_chunk(self, chunk: dict[str, Any]) -> None:
        if not _is_dict(chunk):
            return
        chunk_obj = chunk.get("chunk")
        if _is_dict(chunk_obj):
            raw_bytes = chunk_obj.get("bytes")
            if isinstance(raw_bytes, (bytes, bytearray)):
                self._self_accumulated_text.append(
                    bytes(raw_bytes).decode("utf-8", errors="replace")
                )
            elif isinstance(raw_bytes, str):
                self._self_accumulated_text.append(raw_bytes)

    def _on_stream_end(self) -> None:
        if self._self_capture_content and self._self_accumulated_text:
            content = "".join(self._self_accumulated_text)
            self._self_invocation.output_messages = [
                OutputMessage(
                    role=Role.ASSISTANT.value,
                    parts=[TextPart(content=content)],
                )
            ]
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class BedrockAgentEventStreamWrapper(
    _BedrockAgentEventStreamMixin,
    SyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for Bedrock invoke_agent EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: RemoteAgentInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_agent_stream(invocation, capture_content=capture_content)


class AsyncBedrockAgentEventStreamWrapper(
    _BedrockAgentEventStreamMixin,
    AsyncStreamWrapper[dict[str, Any]],
):
    """Wrapper for async Bedrock invoke_agent EventStream."""

    def __init__(
        self,
        stream: Any,
        invocation: RemoteAgentInvocation,
        *,
        capture_content: bool = True,
    ) -> None:
        super().__init__(stream, invocation=invocation)
        self._init_agent_stream(invocation, capture_content=capture_content)
