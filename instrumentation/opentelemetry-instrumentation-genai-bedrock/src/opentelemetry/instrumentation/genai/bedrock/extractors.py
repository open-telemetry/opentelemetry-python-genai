# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, TypeGuard
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import aws_attributes
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    GenAIInvocation,
    InferenceInvocation,
    RemoteAgentInvocation,
    RetrievalInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericPart,
    GenericToolDefinition,
    InputMessage,
    MessagePart,
    Modality,
    OutputMessage,
    ReasoningPart,
    RetrievalDocument,
    Role,
    SystemInstructionPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import decode_base64


def _is_dict(val: object) -> TypeGuard[dict[str, Any]]:
    return isinstance(val, dict)


def _is_list(val: object) -> TypeGuard[list[Any]]:
    return isinstance(val, list)


def _first_not_none(*values: Any) -> Any:
    """Return the first value that is not None, or None."""
    for val in values:
        if val is not None:
            return val
    return None


_FINISH_REASON_MAP: dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_call",
    "max_tokens": "length",
    "content_filtered": "content_filter",
    "guardrail_intervened": "content_filter",
    "finish": "stop",
    "complete": "stop",
    "endoftext": "stop",
    "length": "length",
    "stop": "stop",
    "tool_calls": "tool_call",
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


def _safe_int(val: Any) -> int | None:
    """Safely convert a value to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _safe_float(val: Any) -> float | None:
    """Safely convert a value to float or return None."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def map_finish_reason(stop_reason: str | None) -> str | None:
    """Map Bedrock stopReason to GenAI semantic convention finish_reason."""
    if stop_reason is None:
        return None
    return _FINISH_REASON_MAP.get(stop_reason.lower(), stop_reason.lower())


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
    """Map a single Bedrock or Anthropic content block to an OpenTelemetry MessagePart."""
    block_type = block.get("type")

    # 1. Text block (Converse or Anthropic)
    if block_type == "text" and "text" in block:
        return TextPart(content=block["text"])
    if "text" in block and block_type is None:
        return TextPart(content=block["text"])

    # 2. Reasoning / thinking block
    reasoning = block.get("reasoningContent")
    if _is_dict(reasoning):
        reasoning_text = reasoning.get("reasoningText")
        if _is_dict(reasoning_text) and "text" in reasoning_text:
            return ReasoningPart(content=reasoning_text["text"])
        if "redactedContent" in reasoning:
            return ReasoningPart(content="")
    if block_type in ("thinking", "redacted_thinking"):
        content = block.get("thinking") or block.get("data") or ""
        return ReasoningPart(content=str(content))

    # 3. Image block
    image = block.get("image")
    if _is_dict(image):
        fmt = image.get("format", "jpeg")
        source = image.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        return BlobPart(
            content=content_bytes,
            mime_type=f"image/{fmt}",
            modality=Modality.IMAGE,
        )
    if block_type == "image":
        source = block.get("source")
        if _is_dict(source):
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/jpeg")
                decoded = decode_base64(source.get("data", ""))
                if decoded is not None:
                    return BlobPart(
                        content=decoded,
                        mime_type=media_type,
                        modality=Modality.IMAGE,
                    )
            elif "bytes" in source:
                return BlobPart(
                    content=source.get("bytes", b""),
                    mime_type=source.get("media_type", "image/jpeg"),
                    modality=Modality.IMAGE,
                )

    # 4. Document block
    document = block.get("document")
    if _is_dict(document):
        fmt = document.get("format", "pdf")
        source = document.get("source")
        content_bytes = source.get("bytes", b"") if _is_dict(source) else b""
        mime_type = _DOC_MIME_TYPES.get(fmt, f"application/{fmt}")
        return BlobPart(
            content=content_bytes,
            mime_type=mime_type,
            modality=Modality.DOCUMENT,
        )
    if block_type == "document":
        source = block.get("source")
        if _is_dict(source):
            if source.get("type") == "base64":
                media_type = source.get("media_type", "application/pdf")
                decoded = decode_base64(source.get("data", ""))
                if decoded is not None:
                    return BlobPart(
                        content=decoded,
                        mime_type=media_type,
                        modality=Modality.DOCUMENT,
                    )
            elif "bytes" in source:
                return BlobPart(
                    content=source.get("bytes", b""),
                    mime_type=source.get("media_type", "application/pdf"),
                    modality=Modality.DOCUMENT,
                )

    # 5. Tool use (Converse toolUse or Anthropic tool_use)
    tool_use = block.get("toolUse")
    if _is_dict(tool_use):
        return ToolCallRequestPart(
            id=tool_use.get("toolUseId"),
            name=tool_use.get("name", ""),
            arguments=tool_use.get("input"),
        )
    if block_type == "tool_use":
        return ToolCallRequestPart(
            id=block.get("id"),
            name=str(block.get("name", "")),
            arguments=block.get("input"),
        )

    # 6. Tool result (Converse toolResult or Anthropic tool_result)
    tool_result = block.get("toolResult")
    if _is_dict(tool_result):
        return ToolCallResponsePart(
            id=tool_result.get("toolUseId"),
            response=tool_result.get("content"),
        )
    if block_type == "tool_result":
        return ToolCallResponsePart(
            id=block.get("tool_use_id"),
            response=block.get("content"),
        )

    # 7. Other Generic block types
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
            return GenericPart(type=key)

    return None


def _extract_parts(content: Any) -> list[MessagePart]:
    if isinstance(content, str):
        return [TextPart(content=content)]
    if not _is_list(content):
        return []
    parts: list[MessagePart] = []
    for item in content:
        if isinstance(item, str):
            parts.append(TextPart(content=item))
        elif _is_dict(item):
            part = extract_content_block(item)
            if part is not None:
                parts.append(part)
    return parts


def _extract_system_parts(
    content: str | Sequence[Mapping[str, Any] | str] | None,
) -> list[SystemInstructionPart]:
    if not content:
        return []
    if isinstance(content, str):
        return [TextPart(content=content)] if content else []
    parts: list[SystemInstructionPart] = []
    for item in content:
        if isinstance(item, str):
            if item:
                parts.append(TextPart(content=item))
        elif _is_dict(item):
            text = item.get("text")
            if isinstance(text, str) and text:
                parts.append(TextPart(content=text))
            else:
                for key in item:
                    if key != "text":
                        parts.append(GenericPart(type=key))
                        break
    return parts


def _extract_guardrail_id(
    params: dict[str, Any], invocation: GenAIInvocation
) -> None:
    guardrail_id = params.get("guardrailIdentifier")
    if not guardrail_id and _is_dict(params.get("guardrailConfig")):
        guardrail_id = params["guardrailConfig"].get("guardrailIdentifier")
    if guardrail_id:
        invocation.attributes[aws_attributes.AWS_BEDROCK_GUARDRAIL_ID] = str(
            guardrail_id
        )


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
        invocation.top_k = _safe_int(
            _first_not_none(inf_config.get("topK"), inf_config.get("top_k"))
        )
        invocation.seed = inf_config.get("seed")

    add_fields = kwargs.get("additionalModelRequestFields")
    if _is_dict(add_fields):
        add_inf = add_fields.get("inferenceConfig")
        top_k_val = _first_not_none(
            add_fields.get("topK"),
            add_fields.get("top_k"),
            add_inf.get("topK") if _is_dict(add_inf) else None,
            add_inf.get("top_k") if _is_dict(add_inf) else None,
            invocation.top_k,
        )
        invocation.top_k = _safe_int(top_k_val)
        if "seed" in add_fields:
            invocation.seed = add_fields.get("seed")

    # Guardrail identifier
    _extract_guardrail_id(kwargs, invocation)

    # Output format
    output_config = kwargs.get("outputConfig")
    if _is_dict(output_config):
        text_format = output_config.get("textFormat")
        if text_format:
            text_format_str = str(text_format).lower()
            if text_format_str in ("json", "text"):
                invocation.output_type = text_format_str

    # Prompt variables (opt-in under content capture)
    prompt_variables = kwargs.get("promptVariables")
    if capture_content and _is_dict(prompt_variables):
        for var_name, var_val in prompt_variables.items():
            if _is_dict(var_val) and "text" in var_val:
                invocation.attributes[f"gen_ai.prompt.variable.{var_name}"] = (
                    str(var_val["text"])
                )

    # system instruction
    raw_system = kwargs.get("system")
    if capture_content and raw_system:
        system_parts = _extract_system_parts(raw_system)
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
            parts = _extract_parts(msg.get("content"))
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
            parts = _extract_parts(msg.get("content"))
            invocation.output_messages = [
                OutputMessage(
                    role=role,
                    parts=parts,
                    finish_reason=finish_reason or "error",
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


def _parse_body(body: Any) -> dict[str, Any] | None:
    """Safely parse body into a dictionary."""
    if _is_dict(body):
        return body
    if isinstance(body, (str, bytes, bytearray)):
        try:
            parsed: object = json.loads(body)
            return parsed if _is_dict(parsed) else None
        except Exception:
            return None
    return None


def extract_invoke_model_request(
    api_params: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate request attributes from InvokeModel api_params onto the invocation."""
    _extract_guardrail_id(api_params, invocation)

    body = _parse_body(api_params.get("body"))
    if not _is_dict(body):
        return

    # Extract optional nested configs
    text_gen_config = (
        body.get("textGenerationConfig")
        if _is_dict(body.get("textGenerationConfig"))
        else None
    )
    inf_config = (
        body.get("inferenceConfig")
        if _is_dict(body.get("inferenceConfig"))
        else None
    )

    # Temperature
    invocation.temperature = _safe_float(
        _first_not_none(
            body.get("temperature"),
            text_gen_config.get("temperature") if text_gen_config else None,
            inf_config.get("temperature") if inf_config else None,
        )
    )

    # Top P
    invocation.top_p = _safe_float(
        _first_not_none(
            body.get("top_p"),
            body.get("topP"),
            body.get("p"),
            text_gen_config.get("topP") if text_gen_config else None,
            inf_config.get("top_p") if inf_config else None,
        )
    )

    # Top K
    invocation.top_k = _safe_int(
        _first_not_none(
            body.get("top_k"),
            body.get("topK"),
            body.get("k"),
            inf_config.get("top_k") if inf_config else None,
        )
    )

    # Max tokens
    invocation.max_tokens = _safe_int(
        _first_not_none(
            body.get("max_tokens"),
            body.get("max_tokens_to_sample"),
            body.get("max_gen_len"),
            body.get("maxTokens"),
            text_gen_config.get("maxTokenCount") if text_gen_config else None,
            inf_config.get("max_new_tokens") if inf_config else None,
        )
    )

    # Stop sequences
    stop_seqs = _first_not_none(
        body.get("stop_sequences"),
        body.get("stopSequences"),
        text_gen_config.get("stopSequences") if text_gen_config else None,
    )
    if _is_list(stop_seqs):
        invocation.stop_sequences = [str(s) for s in stop_seqs]

    # Seed
    invocation.seed = _safe_int(body.get("seed"))

    # Tool definitions (e.g. Anthropic format)
    raw_tools = body.get("tools")
    if _is_list(raw_tools):
        tool_defs: list[ToolDefinition] = []
        for tool in raw_tools:
            if not _is_dict(tool):
                continue
            name = tool.get("name", "")
            description = tool.get("description")
            params = tool.get("input_schema") or tool.get("parameters")
            if params is not None:
                tool_defs.append(
                    FunctionToolDefinition(
                        name=name,
                        description=description,
                        parameters=params if _is_dict(params) else {},
                    )
                )
            elif name and "type" in tool:
                tool_defs.append(
                    GenericToolDefinition(name=name, type=tool["type"])
                )
        invocation.tool_definitions = tool_defs

    if not capture_content:
        return

    # System instruction (e.g. Anthropic / Nova)
    raw_system = body.get("system")
    if raw_system:
        system_parts = _extract_system_parts(raw_system)
        if system_parts:
            invocation.system_instruction = system_parts

    # Input messages / prompt
    if "messages" in body and _is_list(body["messages"]):
        input_messages: list[InputMessage] = []
        for msg in body["messages"]:
            if not _is_dict(msg):
                continue
            role = msg.get("role", "user")
            parts = _extract_parts(msg.get("content"))
            input_messages.append(InputMessage(role=role, parts=parts))
        if input_messages:
            invocation.input_messages = input_messages
    elif "prompt" in body and isinstance(body["prompt"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[TextPart(content=body["prompt"])],
            )
        ]
    elif "inputText" in body and isinstance(body["inputText"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[TextPart(content=body["inputText"])],
            )
        ]
    elif "message" in body and isinstance(body["message"], str):
        invocation.input_messages = [
            InputMessage(
                role="user",
                parts=[TextPart(content=body["message"])],
            )
        ]


def extract_invoke_model_response(
    response: dict[str, Any],
    raw_body_bytes: bytes,
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    """Populate response attributes from InvokeModel response."""
    # 1. Token counts from response headers (case-insensitive)
    resp_meta = response.get("ResponseMetadata")
    http_headers = (
        resp_meta.get("HTTPHeaders") if _is_dict(resp_meta) else None
    )
    if _is_dict(http_headers):
        headers_lower: dict[str, str] = {
            str(k).lower(): str(v) for k, v in http_headers.items()
        }
        invocation.input_tokens = _safe_int(
            headers_lower.get("x-amzn-bedrock-input-token-count")
        )
        invocation.output_tokens = _safe_int(
            headers_lower.get("x-amzn-bedrock-output-token-count")
        )

    body = _parse_body(raw_body_bytes)
    if not _is_dict(body):
        return

    # 2. Token counts from payload if not in headers
    usage = body.get("usage")
    if _is_dict(usage):
        if invocation.input_tokens is None:
            invocation.input_tokens = _safe_int(
                _first_not_none(
                    usage.get("input_tokens"), usage.get("inputTokens")
                )
            )
        if invocation.output_tokens is None:
            invocation.output_tokens = _safe_int(
                _first_not_none(
                    usage.get("output_tokens"), usage.get("outputTokens")
                )
            )
        invocation.cache_read_input_tokens = _safe_int(
            _first_not_none(
                usage.get("cache_read_input_tokens"),
                usage.get("cacheReadInputTokens"),
            )
        )
        invocation.cache_creation_input_tokens = _safe_int(
            _first_not_none(
                usage.get("cache_creation_input_tokens"),
                usage.get("cacheWriteInputTokens"),
            )
        )

    if invocation.input_tokens is None and "inputTextTokenCount" in body:
        invocation.input_tokens = _safe_int(body.get("inputTextTokenCount"))

    results = body.get("results")
    if _is_list(results) and results and _is_dict(results[0]):
        if invocation.output_tokens is None:
            invocation.output_tokens = _safe_int(results[0].get("tokenCount"))

    if invocation.input_tokens is None and "prompt_token_count" in body:
        invocation.input_tokens = _safe_int(body.get("prompt_token_count"))
    if invocation.output_tokens is None and "generation_token_count" in body:
        invocation.output_tokens = _safe_int(
            body.get("generation_token_count")
        )

    # 3. Finish reasons
    raw_finish_reason: str | None = None
    if "stop_reason" in body and isinstance(body["stop_reason"], str):
        raw_finish_reason = body["stop_reason"]
    elif "stopReason" in body and isinstance(body["stopReason"], str):
        raw_finish_reason = body["stopReason"]
    elif _is_list(results) and results and _is_dict(results[0]):
        raw_finish_reason = results[0].get("completionReason")
    elif (
        "outputs" in body
        and _is_list(body["outputs"])
        and body["outputs"]
        and _is_dict(body["outputs"][0])
    ):
        raw_finish_reason = body["outputs"][0].get("stop_reason")
    elif (
        "generations" in body
        and _is_list(body["generations"])
        and body["generations"]
        and _is_dict(body["generations"][0])
    ):
        raw_finish_reason = body["generations"][0].get("finish_reason")
    elif (
        "completions" in body
        and _is_list(body["completions"])
        and body["completions"]
        and _is_dict(body["completions"][0])
    ):
        finish_obj = body["completions"][0].get("finishReason")
        if _is_dict(finish_obj):
            raw_finish_reason = finish_obj.get("reason")

    finish_reason = map_finish_reason(raw_finish_reason)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]

    # Response ID (e.g. Anthropic msg_...)
    if "id" in body and isinstance(body["id"], str):
        invocation.response_id = body["id"]

    # 4. Content capture
    if not capture_content:
        return

    # Anthropic Messages format
    if "content" in body and _is_list(body["content"]):
        parts = _extract_parts(body["content"])
        role = body.get("role", "assistant")
        invocation.output_messages = [
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=finish_reason or "error",
            )
        ]
    # Amazon Nova format
    elif (
        "output" in body
        and _is_dict(body["output"])
        and _is_dict(body["output"].get("message"))
    ):
        msg = body["output"]["message"]
        role = msg.get("role", "assistant")
        nova_parts = _extract_parts(msg.get("content"))
        invocation.output_messages = [
            OutputMessage(
                role=role,
                parts=nova_parts,
                finish_reason=finish_reason or "error",
            )
        ]
    # Anthropic Legacy completion
    elif "completion" in body and isinstance(body["completion"], str):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=body["completion"])],
                finish_reason=finish_reason or "error",
            )
        ]
    # Titan outputText
    elif (
        _is_list(results)
        and results
        and _is_dict(results[0])
        and "outputText" in results[0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=str(results[0]["outputText"]))],
                finish_reason=finish_reason or "error",
            )
        ]
    # Llama generation
    elif "generation" in body and isinstance(body["generation"], str):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=body["generation"])],
                finish_reason=finish_reason or "error",
            )
        ]
    # Mistral outputs
    elif (
        "outputs" in body
        and _is_list(body["outputs"])
        and body["outputs"]
        and _is_dict(body["outputs"][0])
        and "text" in body["outputs"][0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=str(body["outputs"][0]["text"]))],
                finish_reason=finish_reason or "error",
            )
        ]
    # Cohere generations
    elif (
        "generations" in body
        and _is_list(body["generations"])
        and body["generations"]
        and _is_dict(body["generations"][0])
        and "text" in body["generations"][0]
    ):
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=str(body["generations"][0]["text"]))],
                finish_reason=finish_reason or "error",
            )
        ]
    # AI21 completions
    elif (
        "completions" in body
        and _is_list(body["completions"])
        and body["completions"]
        and _is_dict(body["completions"][0])
    ):
        data = body["completions"][0].get("data")
        if _is_dict(data) and "text" in data:
            invocation.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[TextPart(content=str(data["text"]))],
                    finish_reason=finish_reason or "error",
                )
            ]


def is_embedding_model(model_id: str | None) -> bool:
    """Return True if the model ID corresponds to an embedding model.

    Every Bedrock embedding model ID contains ``embed`` (``amazon.titan-embed-*``,
    ``cohere.embed-*``, ``twelvelabs.*-embed-*``), and no text generation model does. The
    match is deliberately broad so that new embedding models are not reported as chat
    completions.
    """
    if not model_id:
        return False
    return "embed" in model_id.lower()


def extract_embedding_request(
    api_params: dict[str, Any],
    invocation: EmbeddingInvocation,
) -> None:
    """Populate request attributes from InvokeModel api_params onto the embedding invocation."""
    _extract_guardrail_id(api_params, invocation)

    body = _parse_body(api_params.get("body"))
    if not _is_dict(body):
        return

    # Dimensions (e.g. Titan Text v2 dimensions or embeddingConfig.outputEmbeddingLength)
    emb_config = body.get("embeddingConfig")
    emb_config_dict = emb_config if _is_dict(emb_config) else None
    dimensions = _safe_int(
        _first_not_none(
            body.get("dimensions"),
            emb_config_dict.get("outputEmbeddingLength")
            if emb_config_dict
            else None,
        )
    )
    if dimensions is not None:
        invocation.dimension_count = dimensions

    # Encoding formats (e.g. Cohere embedding_types)
    raw_formats = _first_not_none(
        body.get("embedding_types"),
        body.get("embeddingTypes"),
        body.get("encoding_format"),
        body.get("encoding_formats"),
    )
    if _is_list(raw_formats):
        formats = [str(fmt) for fmt in raw_formats if fmt is not None]
        if formats:
            invocation.encoding_formats = formats
    elif isinstance(raw_formats, str) and raw_formats:
        invocation.encoding_formats = [raw_formats]


def extract_embedding_response(
    response: dict[str, Any],
    raw_body_bytes: bytes,
    invocation: EmbeddingInvocation,
) -> None:
    """Populate response attributes from InvokeModel embedding response."""
    # 1. Token counts and response model from response headers (case-insensitive)
    resp_meta = response.get("ResponseMetadata")
    http_headers = (
        resp_meta.get("HTTPHeaders") if _is_dict(resp_meta) else None
    )
    if _is_dict(http_headers):
        headers_lower: dict[str, str] = {
            str(k).lower(): str(v) for k, v in http_headers.items()
        }
        if invocation.input_tokens is None:
            invocation.input_tokens = _safe_int(
                headers_lower.get("x-amzn-bedrock-input-token-count")
            )
        model_header = headers_lower.get("x-amzn-bedrock-model-id")
        if model_header and invocation.response_model_name is None:
            invocation.response_model_name = model_header

    body = _parse_body(raw_body_bytes)
    if not _is_dict(body):
        return

    # 2. Token counts from payload if not in headers
    if invocation.input_tokens is None:
        token_count = body.get("inputTextTokenCount")
        if token_count is None and _is_dict(body.get("meta")):
            meta = body["meta"]
            if _is_dict(meta.get("billed_units")):
                token_count = meta["billed_units"].get("input_tokens")
        invocation.input_tokens = _safe_int(token_count)

    # 3. Dimension count from embeddings in body
    embedding = body.get("embedding")
    if _is_list(embedding):
        invocation.dimension_count = len(embedding)
    else:
        embeddings = body.get("embeddings")
        if _is_list(embeddings) and embeddings:
            first = embeddings[0]
            if _is_list(first):
                invocation.dimension_count = len(first)
        elif _is_dict(embeddings):
            for emb_list in embeddings.values():
                if _is_list(emb_list) and emb_list:
                    first = emb_list[0]
                    if _is_list(first):
                        invocation.dimension_count = len(first)
                        break


def extract_invoke_agent_request(
    api_params: dict[str, Any],
    invocation: RemoteAgentInvocation,
    *,
    capture_content: bool = True,
) -> None:
    session_id = api_params.get("sessionId")
    if session_id:
        invocation.conversation_id = str(session_id)

    if capture_content:
        input_text = api_params.get("inputText")
        if input_text is not None:
            invocation.input_messages = [
                InputMessage(
                    role=Role.USER.value,
                    parts=[TextPart(content=str(input_text))],
                )
            ]


def _extract_functions(ag: dict[str, Any]) -> list[dict[str, Any]]:
    schema = ag.get("functionSchema")
    if not _is_dict(schema):
        return []
    functions = schema.get("functions")
    if not _is_list(functions):
        return []
    return [f for f in functions if _is_dict(f)]


def extract_invoke_inline_agent_request(
    api_params: dict[str, Any],
    invocation: RemoteAgentInvocation,
    *,
    capture_content: bool = True,
) -> None:
    session_id = api_params.get("sessionId")
    if session_id:
        invocation.conversation_id = str(session_id)

    guardrail_cfg = api_params.get("guardrailConfiguration")
    if _is_dict(guardrail_cfg):
        guardrail_id = guardrail_cfg.get(
            "guardrailIdentifier"
        ) or guardrail_cfg.get("guardrailId")
        if guardrail_id:
            invocation.attributes[aws_attributes.AWS_BEDROCK_GUARDRAIL_ID] = (
                str(guardrail_id)
            )

    kbs = api_params.get("knowledgeBases")
    if _is_list(kbs) and kbs and _is_dict(kbs[0]):
        kb_id = kbs[0].get("knowledgeBaseId")
        if kb_id:
            invocation.attributes[
                aws_attributes.AWS_BEDROCK_KNOWLEDGE_BASE_ID
            ] = str(kb_id)

    action_groups = api_params.get("actionGroups")
    if _is_list(action_groups):
        tool_defs: list[ToolDefinition] = []
        for ag in action_groups:
            if not _is_dict(ag):
                continue
            for func in _extract_functions(ag):
                name = str(func.get("name", ""))
                desc = func.get("description")
                params = func.get("parameters")
                tool_defs.append(
                    FunctionToolDefinition(
                        name=name,
                        description=str(desc) if desc is not None else None,
                        parameters=params if _is_dict(params) else {},
                    )
                )
            sig = ag.get("parentActionGroupSignature")
            if sig:
                tool_defs.append(
                    GenericToolDefinition(
                        name=ag.get("actionGroupName", str(sig)),
                        type=str(sig),
                    )
                )
        if tool_defs:
            invocation.tool_definitions = tool_defs

    if capture_content:
        instruction = api_params.get("instruction")
        if instruction is not None:
            invocation.system_instruction = [
                TextPart(content=str(instruction))
            ]

        input_text = api_params.get("inputText")
        if input_text is not None:
            invocation.input_messages = [
                InputMessage(
                    role=Role.USER.value,
                    parts=[TextPart(content=str(input_text))],
                )
            ]


def extract_invoke_inline_agent_model(
    api_params: dict[str, Any],
) -> str | None:
    raw_model = api_params.get("foundationModel")
    if not raw_model:
        return None
    return _extract_model_from_arn(raw_model) or str(raw_model)


def extract_retrieve_request(
    api_params: dict[str, Any],
    invocation: RetrievalInvocation,
    *,
    capture_content: bool = True,
) -> None:
    raw_kb_id = api_params.get("knowledgeBaseId")
    if raw_kb_id:
        invocation.attributes[aws_attributes.AWS_BEDROCK_KNOWLEDGE_BASE_ID] = (
            str(raw_kb_id)
        )

    retrieval_config = api_params.get("retrievalConfiguration")
    if _is_dict(retrieval_config):
        vector_search_config = retrieval_config.get(
            "vectorSearchConfiguration"
        )
        if _is_dict(vector_search_config):
            top_k = vector_search_config.get("numberOfResults")
            if top_k is not None:
                invocation.top_k = _safe_int(top_k)

    if capture_content:
        retrieval_query = api_params.get("retrievalQuery")
        if _is_dict(retrieval_query):
            query_text = retrieval_query.get("text")
            if query_text is not None:
                invocation.query_text = str(query_text)


def _extract_model_from_arn(model_arn: Any) -> str | None:
    if not model_arn:
        return None
    arn_str = str(model_arn)
    for prefix in ("foundation-model/", "inference-profile/"):
        if prefix in arn_str:
            return arn_str.split(prefix, 1)[1]
    return arn_str


def extract_retrieve_and_generate_model(
    api_params: dict[str, Any],
) -> str | None:
    rag_config = api_params.get("retrieveAndGenerateConfiguration")
    if not _is_dict(rag_config):
        return None
    for sub_key in (
        "knowledgeBaseConfiguration",
        "externalSourcesConfiguration",
    ):
        sub_cfg = rag_config.get(sub_key)
        if _is_dict(sub_cfg) and sub_cfg.get("modelArn"):
            return _extract_model_from_arn(sub_cfg.get("modelArn"))
    return None


def extract_retrieve_and_generate_request(
    api_params: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    session_id = api_params.get("sessionId")
    if session_id:
        invocation.conversation_id = str(session_id)

    if capture_content:
        input_obj = api_params.get("input")
        if _is_dict(input_obj):
            input_text = input_obj.get("text")
            if input_text is not None:
                invocation.input_messages = [
                    InputMessage(
                        role=Role.USER.value,
                        parts=[TextPart(content=str(input_text))],
                    )
                ]

    rag_config = api_params.get("retrieveAndGenerateConfiguration")
    if not _is_dict(rag_config):
        return

    gen_config: dict[str, Any] | None = None
    kb_config = rag_config.get("knowledgeBaseConfiguration")
    if _is_dict(kb_config):
        kb_id = kb_config.get("knowledgeBaseId")
        if kb_id:
            invocation.attributes[
                aws_attributes.AWS_BEDROCK_KNOWLEDGE_BASE_ID
            ] = str(kb_id)

        if _is_dict(kb_config.get("generationConfiguration")):
            gen_config = kb_config["generationConfiguration"]

    ext_config = rag_config.get("externalSourcesConfiguration")
    if (
        gen_config is None
        and _is_dict(ext_config)
        and _is_dict(ext_config.get("generationConfiguration"))
    ):
        gen_config = ext_config["generationConfiguration"]

    if _is_dict(gen_config):
        guardrail_cfg = gen_config.get("guardrailConfiguration")
        if _is_dict(guardrail_cfg):
            guardrail_id = guardrail_cfg.get("guardrailId")
            if guardrail_id:
                invocation.attributes[
                    aws_attributes.AWS_BEDROCK_GUARDRAIL_ID
                ] = str(guardrail_id)

        inf_cfg = gen_config.get("inferenceConfig")
        if _is_dict(inf_cfg):
            text_inf = inf_cfg.get("textInferenceConfig")
            if _is_dict(text_inf):
                if "temperature" in text_inf:
                    invocation.temperature = _safe_float(
                        text_inf.get("temperature")
                    )
                if "topP" in text_inf:
                    invocation.top_p = _safe_float(text_inf.get("topP"))
                if "maxTokens" in text_inf:
                    invocation.max_tokens = _safe_int(
                        text_inf.get("maxTokens")
                    )
                stop_seq = text_inf.get("stopSequences")
                if _is_list(stop_seq):
                    invocation.stop_sequences = [
                        str(s) for s in stop_seq if s is not None
                    ]

        add_fields = gen_config.get("additionalModelRequestFields")
        if _is_dict(add_fields):
            top_k = _first_not_none(
                add_fields.get("topK"), add_fields.get("top_k")
            )
            if top_k is not None:
                invocation.top_k = _safe_int(top_k)
            if invocation.seed is None and "seed" in add_fields:
                invocation.seed = _safe_int(add_fields.get("seed"))


def extract_retrieve_and_generate_response(
    response: dict[str, Any],
    invocation: InferenceInvocation,
    *,
    capture_content: bool = True,
) -> None:
    session_id = response.get("sessionId")
    if session_id and not invocation.conversation_id:
        invocation.conversation_id = str(session_id)

    guardrail_action = response.get("guardrailAction")
    finish_reason = (
        "content_filter" if guardrail_action == "INTERVENED" else "stop"
    )
    invocation.finish_reasons = [finish_reason]

    if not capture_content:
        return

    output = response.get("output")
    if _is_dict(output):
        text = output.get("text")
        if text is not None:
            invocation.output_messages = [
                OutputMessage(
                    role=Role.ASSISTANT.value,
                    parts=[TextPart(content=str(text))],
                    finish_reason=finish_reason,
                )
            ]


def extract_retrieve_response(
    response: dict[str, Any],
    invocation: RetrievalInvocation,
    *,
    capture_content: bool = True,
) -> None:
    if not capture_content:
        return

    results = response.get("retrievalResults")
    if not _is_list(results):
        return

    docs: list[RetrievalDocument] = []
    for item in results:
        if not _is_dict(item):
            continue

        document_id: str | None = None
        raw_doc_id = item.get("documentId")
        if raw_doc_id is not None:
            document_id = str(raw_doc_id)

        # `location` is a union keyed by data source (s3Location, webLocation, …), each
        # holding a single `uri`/`url` member. Scanning generically keeps new AWS data
        # source types working; sqlLocation has no locator and is skipped.
        location = item.get("location")
        if document_id is None and _is_dict(location):
            for key, value in location.items():
                if key == "type" or not _is_dict(value):
                    continue
                locator = _first_not_none(
                    value.get("uri"), value.get("url"), value.get("id")
                )
                if locator is not None:
                    document_id = str(locator)
                    break

        score = _safe_float(item.get("score"))

        if document_id is not None or score is not None:
            docs.append(RetrievalDocument(id=document_id, score=score))

    if docs:
        invocation.documents = docs
