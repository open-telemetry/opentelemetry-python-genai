# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
#
# Based on the Bedrock extension in opentelemetry-python-contrib by @xrmx:
# https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3161
# https://github.com/open-telemetry/opentelemetry-python-contrib/pull/3258

"""Get/extract helpers for AWS Bedrock Converse instrumentation."""

from __future__ import annotations

from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)
from opentelemetry.util.types import AttributeValue


def normalize_finish_reason(stop_reason: str | None) -> str | None:
    """Normalize Bedrock stop reasons to semconv values."""
    if stop_reason is None:
        return None
    normalized = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }.get(stop_reason)
    return normalized or stop_reason


def get_request_attributes(
    api_params: dict[str, Any],
) -> dict[str, AttributeValue]:
    """Extract GenAI request attributes from Converse API parameters."""
    inference_config = api_params.get("inferenceConfig") or {}

    attributes: dict[str, AttributeValue | None] = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_SYSTEM: GenAIAttributes.GenAiSystemValues.AWS_BEDROCK.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: api_params.get("modelId"),
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: inference_config.get(
            "maxTokens"
        ),
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: inference_config.get(
            "temperature"
        ),
        GenAIAttributes.GEN_AI_REQUEST_TOP_P: inference_config.get("topP"),
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: inference_config.get(
            "stopSequences"
        ),
    }
    return {k: v for k, v in attributes.items() if v is not None}


def get_response_attributes(
    result: dict[str, Any],
) -> dict[str, Any]:
    """Extract response attributes from a Converse result dict."""
    attrs: dict[str, Any] = {}

    # Stop reason
    stop_reason = result.get("stopReason")
    finish_reason = normalize_finish_reason(stop_reason)
    if finish_reason:
        attrs["finish_reasons"] = [finish_reason]

    # Usage tokens
    usage = result.get("usage") or {}
    input_tokens = usage.get("inputTokens")
    output_tokens = usage.get("outputTokens")
    if input_tokens is not None:
        attrs["input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["output_tokens"] = output_tokens

    # Response metadata
    response_metadata = result.get("ResponseMetadata") or {}
    request_id = response_metadata.get("RequestId")
    if request_id:
        attrs["response_id"] = request_id

    # Bedrock does not return a response model name in the standard Converse
    # response. The model used is the one requested.
    attrs["response_model"] = None

    return attrs


def get_input_messages(
    api_params: dict[str, Any],
) -> list[InputMessage]:
    """Extract input messages from Converse API parameters."""
    messages = api_params.get("messages") or []
    result: list[InputMessage] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or []
        parts = _convert_content_blocks_to_parts(content)
        result.append(InputMessage(role=role, parts=parts))
    return result


def get_system_instruction(
    api_params: dict[str, Any],
) -> list[MessagePart]:
    """Extract system instruction from Converse API parameters."""
    system = api_params.get("system") or []
    parts: list[MessagePart] = []
    for block in system:
        text = block.get("text")
        if text is not None:
            parts.append(Text(content=str(text)))
    return parts


def get_output_messages(
    result: dict[str, Any],
) -> list[OutputMessage]:
    """Extract output messages from a Converse response."""
    output = result.get("output") or {}
    message = output.get("message")
    if message is None:
        return []

    role = message.get("role", "assistant")
    content = message.get("content") or []
    parts = _convert_content_blocks_to_parts(content)

    stop_reason = result.get("stopReason")
    finish_reason = normalize_finish_reason(stop_reason) or ""

    return [OutputMessage(role=role, parts=parts, finish_reason=finish_reason)]


def _convert_content_blocks_to_parts(
    content: list[dict[str, Any]],
) -> list[MessagePart]:
    """Convert Bedrock content blocks to MessagePart instances."""
    parts: list[MessagePart] = []
    for block in content:
        part = _convert_block_to_part(block)
        if part is not None:
            parts.append(part)
    return parts


def _convert_block_to_part(block: dict[str, Any]) -> MessagePart | None:
    """Convert a single Bedrock content block to a MessagePart."""
    # Text block
    if "text" in block:
        return Text(content=str(block["text"]))

    # Tool use block (request)
    if "toolUse" in block:
        tool_use = block["toolUse"]
        return ToolCallRequest(
            arguments=tool_use.get("input"),
            name=tool_use.get("name", ""),
            id=tool_use.get("toolUseId"),
        )

    # Tool result block (response)
    if "toolResult" in block:
        tool_result = block["toolResult"]
        content_parts = tool_result.get("content") or []
        # Flatten text content from tool result
        response_text = ""
        for part in content_parts:
            if "text" in part:
                response_text += part["text"]
        return ToolCallResponse(
            response=response_text or tool_result.get("content"),
            id=tool_result.get("toolUseId"),
        )

    return None
