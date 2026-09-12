# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for DSPy instrumentation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeGuard

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GenAiProviderNameValues,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FunctionToolDefinition,
    GenericPart,
    InputMessage,
    MessagePart,
    Modality,
    OutputMessage,
    ReasoningPart,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
    UriPart,
)
from opentelemetry.util.genai.utils import decode_base64, image_from_url

if TYPE_CHECKING:
    from dspy.adapters.types.tool import Tool
    from dspy.clients.base_lm import BaseLM
    from dspy.core.types import LMBasePart, LMMessage, LMResponse
    from dspy.primitives.prediction import Prediction

    from opentelemetry.util.genai.invocation import InferenceInvocation

SENTINEL_TOOL_NAMES: frozenset[str] = frozenset({"finish", "submit"})

_KNOWN_PROVIDERS: dict[str, str] = {
    "openai": GenAiProviderNameValues.OPENAI.value,
    "anthropic": GenAiProviderNameValues.ANTHROPIC.value,
    "cohere": GenAiProviderNameValues.COHERE.value,
    "bedrock": GenAiProviderNameValues.AWS_BEDROCK.value,
    "vertex_ai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "vertexai": GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "gemini": GenAiProviderNameValues.GCP_GEMINI.value,
    "google": GenAiProviderNameValues.GCP_GEMINI.value,
    "groq": GenAiProviderNameValues.GROQ.value,
    "mistral": GenAiProviderNameValues.MISTRAL_AI.value,
    "deepseek": GenAiProviderNameValues.DEEPSEEK.value,
    "azure": GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "watsonx": GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "perplexity": GenAiProviderNameValues.PERPLEXITY.value,
    "xai": GenAiProviderNameValues.X_AI.value,
}


def _is_mapping(val: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(val, Mapping)


def _is_sequence(val: object) -> TypeGuard[Sequence[object]]:
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def safe_int(val: object) -> int | None:
    """Safely convert a value to int or return None."""
    if isinstance(val, (int, float, str, bytes)):
        try:
            return int(val)
        except (ValueError, TypeError):
            return None
    return None


def safe_float(val: object) -> float | None:
    """Safely convert a value to float or return None."""
    if isinstance(val, (int, float, str, bytes)):
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return None


def safe_stop_sequences(val: object) -> list[str] | None:
    """Extract stop sequences as a list of strings."""
    if isinstance(val, str):
        return [val]
    if _is_sequence(val):
        return [str(s) for s in val]
    return None


_safe_int = safe_int
_safe_float = safe_float
_safe_stop_sequences = safe_stop_sequences


def parse_provider_and_model(
    model_str: str | None,
) -> tuple[str | None, str | None]:
    """Parse provider and model name from LiteLLM-style model string."""
    if not isinstance(model_str, str) or not model_str:
        return None, None
    model_str = model_str.strip().rstrip("/")
    if "/" in model_str:
        provider, model_name = model_str.split("/", 1)
        provider = provider.strip().lower()
        model_name = model_name.strip()
        if "-" in provider:
            provider = provider.split("-")[-1]
        return provider, model_name
    return None, model_str.strip() if model_str else None


def resolve_provider(instance: BaseLM) -> str:
    """Resolve gen_ai.provider.name from a DSPy LM instance."""
    provider_obj = getattr(instance, "provider", None)
    if provider_obj is not None:
        cls_name = provider_obj.__class__.__name__.lower()
        if "provider" in cls_name and cls_name != "provider":
            p_name = cls_name.removesuffix("provider")
            return _KNOWN_PROVIDERS.get(p_name) or p_name

    model = getattr(instance, "model", None)
    if model is not None:
        p_name, _ = parse_provider_and_model(str(model))
        if p_name:
            return _KNOWN_PROVIDERS.get(p_name) or p_name
        model_str = str(model).lower()
        if model_str.startswith("gemini"):
            return GenAiProviderNameValues.GCP_GEMINI.value
        if model_str.startswith("claude"):
            return GenAiProviderNameValues.ANTHROPIC.value
        if model_str.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
            return GenAiProviderNameValues.OPENAI.value

    if instance.__class__.__name__ == "DummyLM" or model == "dummy":
        return "dummy"

    return "unknown"


def resolve_request_model(instance: BaseLM) -> str | None:
    """Resolve model name from a DSPy LM instance."""
    if (model_name := getattr(instance, "model_name", None)) is not None:
        return str(model_name)
    model = getattr(instance, "model", None)
    if model is not None:
        _, m_name = parse_provider_and_model(str(model))
        return m_name or str(model)
    return None


def _extract_tool_call_part(
    tc: object,
    capture_content: bool = True,
) -> ToolCallRequestPart | None:
    """Extract a ToolCallRequestPart from a tool call dictionary or object."""
    if _is_mapping(tc):
        call_id = tc.get("id")
        func = tc.get("function")
        if _is_mapping(func):
            name = str(func.get("name", ""))
            args_raw = func.get("arguments")
        else:
            name = str(tc.get("name", ""))
            args_raw = tc.get("args") or tc.get("arguments")

        args = None
        if capture_content:
            args = args_raw
            if isinstance(args_raw, str):
                try:
                    args = json.loads(args_raw)
                except (json.JSONDecodeError, TypeError):
                    args = args_raw
        return ToolCallRequestPart(
            id=str(call_id) if call_id else None,
            name=name,
            arguments=args,
        )
    if hasattr(tc, "name"):
        call_id = getattr(tc, "id", None)
        name = str(getattr(tc, "name", ""))
        args = None
        if capture_content:
            args = getattr(tc, "args", None) or getattr(tc, "arguments", None)
        return ToolCallRequestPart(
            id=str(call_id) if call_id else None,
            name=name,
            arguments=args,
        )
    return None


def _derive_modality(
    p_type: str | None, media_type: str | None
) -> Modality | str:
    if p_type in ("image", "image_url"):
        return "image"
    if p_type in ("audio", "input_audio"):
        return "audio"
    if p_type == "video":
        return "video"
    if p_type == "document":
        return "document"
    if media_type and "/" in media_type:
        prefix = media_type.split("/")[0]
        if prefix in ("image", "video", "audio", "document"):
            return prefix
    return "document"


def _to_bytes(data: object) -> bytes | None:
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return decode_base64(data)
    return None


def _extract_part(
    p: LMBasePart | Mapping[str, object] | str | object,
    capture_content: bool = True,
) -> MessagePart | None:
    """Extract a MessagePart from a part object or dictionary."""
    if isinstance(p, str):
        return TextPart(content=p)

    if _is_mapping(p):
        p_type = p.get("type")
        p_type_str = str(p_type) if p_type is not None else None

        if p_type_str == "text" or ("text" in p and not p_type):
            content = p.get("text") or p.get("content")
            if isinstance(content, str):
                return TextPart(content=content)
        if p_type_str in ("thinking", "reasoning") or (
            "reasoning" in p and not p_type
        ):
            content = p.get("thinking") or p.get("reasoning") or p.get("text")
            if isinstance(content, str):
                return ReasoningPart(content=content)
        if p_type_str == "refusal":
            return GenericPart(type="refusal")
        if p_type_str == "tool_call" or "function" in p:
            return _extract_tool_call_part(p, capture_content=capture_content)
        if (
            p_type_str == "tool_result"
            or "tool_call_id" in p
            or p.get("role") == "tool"
        ):
            call_id = p.get("tool_call_id") or p.get("call_id") or p.get("id")
            content = p.get("content") or p.get("response")
            return ToolCallResponsePart(
                id=str(call_id) if call_id else None,
                response=content,
            )

        # URL specified -> UriPart
        url = p.get("url") or p.get("path")
        if not url and p_type_str == "image_url":
            img_url = p.get("image_url")
            if _is_mapping(img_url):
                url = img_url.get("url")
            elif isinstance(img_url, str):
                url = img_url

        media_type = p.get("media_type") or p.get("mime_type")
        media_type_str = str(media_type) if media_type else None

        if url:
            modality = _derive_modality(p_type_str, media_type_str)
            if isinstance(url, str) and url.startswith("data:"):
                part = image_from_url(url, modality=modality)
                if part:
                    return part
            return UriPart(
                mime_type=media_type_str,
                modality=modality,
                uri=str(url),
            )

        # Inline data
        data = p.get("data")
        if data is None and p_type_str in ("audio", "input_audio"):
            data = p.get("input_audio")

        if p_type_str in ("audio", "video") and data is not None:
            # Inline audio and video payloads are omitted from telemetry spans
            # because they can be massive.
            return GenericPart(type=p_type_str)

        if data is not None:
            content_bytes = _to_bytes(data)
            if content_bytes is not None:
                modality = _derive_modality(p_type_str, media_type_str)
                return BlobPart(
                    mime_type=media_type_str,
                    modality=modality,
                    content=content_bytes,
                )

        # Fallback to GenericPart for any unhandled dictionary part
        return GenericPart(type=p_type_str or "custom")

    # Object handling (DSPy LMBasePart subclasses, etc.)
    p_type = getattr(p, "type", None)
    p_type_str = str(p_type) if p_type is not None else None

    if p_type_str == "text":
        text = getattr(p, "text", None)
        if isinstance(text, str):
            return TextPart(content=text)
    elif p_type_str == "thinking":
        text = getattr(p, "text", None)
        if isinstance(text, str):
            return ReasoningPart(content=text)
    elif p_type_str == "refusal":
        return GenericPart(type="refusal")
    elif p_type_str == "tool_call" or (
        hasattr(p, "args") and hasattr(p, "name")
    ):
        return _extract_tool_call_part(p, capture_content=capture_content)
    elif p_type_str == "tool_result" or hasattr(p, "call_id"):
        call_id = getattr(p, "call_id", None)
        content = getattr(p, "content", None)
        return ToolCallResponsePart(
            id=str(call_id) if call_id else None,
            response=content,
        )

    # URL specified -> UriPart
    url = getattr(p, "url", None) or getattr(p, "path", None)
    media_type = getattr(p, "media_type", None)
    media_type_str = str(media_type) if media_type else None

    if url:
        modality = _derive_modality(p_type_str, media_type_str)
        if isinstance(url, str) and url.startswith("data:"):
            part = image_from_url(url, modality=modality)
            if part:
                return part
        return UriPart(
            mime_type=media_type_str,
            modality=modality,
            uri=str(url),
        )

    # Inline data
    data = getattr(p, "data", None)
    if p_type_str in ("audio", "video") and data is not None:
        # Inline audio and video payloads are omitted from telemetry spans
        # because their large size causes excessive memory overhead and span bloat.
        return GenericPart(type=p_type_str)

    if data is not None:
        content_bytes = _to_bytes(data)
        if content_bytes is not None:
            modality = _derive_modality(p_type_str, media_type_str)
            return BlobPart(
                mime_type=media_type_str,
                modality=modality,
                content=content_bytes,
            )

    if p_type_str is None:
        text = getattr(p, "text", None)
        if isinstance(text, str):
            return TextPart(content=text)

    # Fallback to GenericPart for any other part that we haven't covered
    fallback_type = p_type_str or type(p).__name__
    return GenericPart(type=fallback_type)


def _extract_single_message(
    msg: LMMessage | Mapping[str, object] | object,
    capture_content: bool = True,
) -> InputMessage | None:
    if _is_mapping(msg):
        role = str(msg.get("role", "user"))
        name = msg.get("name")
        parts: list[MessagePart] = []

        if role == "tool" or "tool_call_id" in msg:
            call_id = msg.get("tool_call_id") or msg.get("id")
            content = msg.get("content")
            parts.append(
                ToolCallResponsePart(
                    id=str(call_id) if call_id else None,
                    response=content,
                )
            )
        else:
            tool_calls = msg.get("tool_calls")
            if _is_sequence(tool_calls):
                for tc in tool_calls:
                    tcp = _extract_tool_call_part(
                        tc, capture_content=capture_content
                    )
                    if tcp:
                        parts.append(tcp)

            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                parts.append(ReasoningPart(content=reasoning))

            raw_parts = msg.get("parts")
            if _is_sequence(raw_parts):
                for p in raw_parts:
                    extracted_p = _extract_part(
                        p, capture_content=capture_content
                    )
                    if extracted_p:
                        parts.append(extracted_p)

            content = msg.get("content")
            if _is_sequence(content):
                for p in content:
                    extracted_p = _extract_part(
                        p, capture_content=capture_content
                    )
                    if extracted_p:
                        parts.append(extracted_p)
            elif isinstance(content, str):
                parts.append(TextPart(content=content))
            elif content is not None and not parts:
                parts.append(TextPart(content=str(content)))

        if parts:
            return InputMessage(
                role=role,
                parts=parts,
                name=str(name) if name is not None else None,
            )
        return None

    role = getattr(msg, "role", None)
    if role is None:
        return None
    name = getattr(msg, "name", None)
    parts: list[MessagePart] = []
    raw_msg_parts = getattr(msg, "parts", None)
    if _is_sequence(raw_msg_parts):
        for p in raw_msg_parts:
            extracted = _extract_part(p, capture_content=capture_content)
            if extracted:
                parts.append(extracted)

    msg_text = getattr(msg, "text", None)
    if not parts and isinstance(msg_text, str) and msg_text:
        parts.append(TextPart(content=msg_text))

    if parts:
        return InputMessage(
            role=str(role),
            parts=parts,
            name=str(name) if name is not None else None,
        )

    return None


def extract_lm_input_messages(
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    capture_content: bool = True,
) -> list[InputMessage]:
    """Extract InputMessage list from LM call arguments."""
    request = kwargs.get("request")
    if request is None and args and hasattr(args[0], "messages"):
        request = args[0]

    if request is not None and hasattr(request, "messages"):
        raw_req_msgs = getattr(request, "messages", None)
        if _is_sequence(raw_req_msgs):
            msgs: list[InputMessage] = []
            for m in raw_req_msgs:
                extracted = _extract_single_message(
                    m, capture_content=capture_content
                )
                if extracted:
                    msgs.append(extracted)
            if msgs:
                return msgs

    messages = kwargs.get("messages")
    if _is_sequence(messages):
        msgs: list[InputMessage] = []
        for m in messages:
            extracted = _extract_single_message(
                m, capture_content=capture_content
            )
            if extracted:
                msgs.append(extracted)
        if msgs:
            return msgs

    prompt = kwargs.get("prompt")
    if prompt is None and args and isinstance(args[0], str):
        prompt = args[0]

    if prompt is not None:
        return [
            InputMessage(role="user", parts=[TextPart(content=str(prompt))])
        ]

    if args:
        msgs: list[InputMessage] = []
        for a in args:
            extracted = _extract_single_message(
                a, capture_content=capture_content
            )
            if extracted:
                msgs.append(extracted)
        if msgs:
            return msgs

    return []


def extract_lm_output_messages(
    result: LMResponse | Sequence[Mapping[str, object] | str] | str,
    finish_reason: str | None = None,
    capture_content: bool = True,
) -> list[OutputMessage]:
    """Extract OutputMessage list from LM call result."""
    if isinstance(result, str):
        return [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=result)],
                finish_reason=finish_reason,
            )
        ]

    if not isinstance(result, Sequence):
        msgs: list[OutputMessage] = []
        for out in result.outputs:
            parts: list[MessagePart] = []
            if out.parts:
                for p in out.parts:
                    extracted = _extract_part(
                        p, capture_content=capture_content
                    )
                    if extracted:
                        parts.append(extracted)
            if not parts:
                if out.text:
                    parts.append(TextPart(content=out.text))
                if (
                    isinstance(out.reasoning_content, str)
                    and out.reasoning_content
                ):
                    parts.append(ReasoningPart(content=out.reasoning_content))
                for tc in out.tool_calls:
                    tcp = _extract_tool_call_part(
                        tc, capture_content=capture_content
                    )
                    if tcp:
                        parts.append(tcp)
            fr = out.finish_reason or finish_reason
            if parts:
                msgs.append(
                    OutputMessage(
                        role="assistant",
                        parts=parts,
                        finish_reason=str(fr) if fr else None,
                    )
                )
        return msgs

    msgs: list[OutputMessage] = []
    for item in result:
        if isinstance(item, str):
            msgs.append(
                OutputMessage(
                    role="assistant",
                    parts=[TextPart(content=item)],
                    finish_reason=finish_reason,
                )
            )
        else:
            parts: list[MessagePart] = []
            raw_parts = item.get("parts")
            if _is_sequence(raw_parts):
                for p in raw_parts:
                    extracted_p = _extract_part(
                        p, capture_content=capture_content
                    )
                    if extracted_p:
                        parts.append(extracted_p)

            content = item.get("content") or item.get("text")
            if _is_sequence(content):
                for p in content:
                    extracted_p = _extract_part(
                        p, capture_content=capture_content
                    )
                    if extracted_p:
                        parts.append(extracted_p)
            elif isinstance(content, str):
                parts.append(TextPart(content=content))

            reasoning = item.get("reasoning_content") or item.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                parts.append(ReasoningPart(content=reasoning))

            tool_calls = item.get("tool_calls")
            if _is_sequence(tool_calls):
                for tc in tool_calls:
                    tcp = _extract_tool_call_part(
                        tc, capture_content=capture_content
                    )
                    if tcp:
                        parts.append(tcp)

            if not parts:
                parts.append(TextPart(content=str(item)))

            fr = item.get("finish_reason") or finish_reason
            msgs.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=str(fr) if fr else None,
                )
            )
    return msgs


def apply_usage_to_invocation(
    invocation: InferenceInvocation,
    usage: Mapping[str, object],
) -> None:
    """Apply token usage dictionary to an InferenceInvocation."""
    in_tokens = usage.get("prompt_tokens")
    if in_tokens is None:
        in_tokens = usage.get("input_tokens")
    invocation.input_tokens = _safe_int(in_tokens)

    out_tokens = usage.get("completion_tokens")
    if out_tokens is None:
        out_tokens = usage.get("output_tokens")
    invocation.output_tokens = _safe_int(out_tokens)

    invocation.thinking_tokens = _safe_int(usage.get("reasoning_tokens"))
    invocation.cache_read_input_tokens = _safe_int(
        usage.get("cache_read_tokens")
    )
    invocation.cache_write_input_tokens = _safe_int(
        usage.get("cache_write_tokens")
    )


def extract_input_content(input_args: Mapping[str, object]) -> str:
    """Extract input content string from agent invocation arguments."""
    if not input_args:
        return ""
    if len(input_args) == 1:
        val = next(iter(input_args.values()))
        if isinstance(val, str):
            return val
        if isinstance(val, (int, float, bool)):
            return str(val)
        try:
            return json.dumps(val, ensure_ascii=False)
        except Exception:
            return str(val)
    try:
        return json.dumps(input_args, ensure_ascii=False)
    except Exception:
        return str(input_args)


def extract_output_content(
    result: Prediction | None,
    signature: object = None,
) -> str:
    """Extract output content string from a DSPy prediction result."""
    if result is None:
        return ""

    output_dict: dict[str, object] = {}
    if hasattr(result, "items") and callable(getattr(result, "items")):
        try:
            items_func: Callable[[], Iterable[tuple[object, object]]] = (
                getattr(result, "items")
            )
            for k, v in items_func():
                output_dict[str(k)] = v
        except Exception:
            pass

    if output_dict:
        filtered_dict: dict[str, object] = {}
        output_fields_raw = (
            getattr(signature, "output_fields", None) if signature else None
        )
        if _is_sequence(output_fields_raw):
            for key in output_fields_raw:
                key_str = str(key)
                if key_str in output_dict:
                    filtered_dict[key_str] = output_dict[key_str]
        if not filtered_dict:
            for k_str, v_val in output_dict.items():
                if k_str not in (
                    "trajectory",
                    "history",
                    "termination_reason",
                ):
                    filtered_dict[k_str] = v_val

        if filtered_dict:
            if len(filtered_dict) == 1:
                val = next(iter(filtered_dict.values()))
                if isinstance(val, str):
                    return val
                if isinstance(val, (int, float, bool)):
                    return str(val)
                try:
                    return json.dumps(val, ensure_ascii=False)
                except Exception:
                    return str(val)
            try:
                return json.dumps(filtered_dict, ensure_ascii=False)
            except Exception:
                return str(filtered_dict)

    return str(result)


def prepare_tool_definitions(
    tools: Sequence[Tool | Callable[..., object]]
    | Mapping[str, Tool | Callable[..., object]]
    | None,
) -> list[ToolDefinition] | None:
    """Prepare FunctionToolDefinition instances from a tools collection."""
    if not tools:
        return None

    if isinstance(tools, Mapping):
        tool_items = list(tools.values())
    elif isinstance(tools, (list, tuple, set)):
        tool_items = list(tools)
    else:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tool_items:
        tool_name = getattr(tool, "name", None)
        if not tool_name:
            func = getattr(tool, "func", None)
            tool_name = (
                getattr(func, "__name__", None)
                if func
                else getattr(tool, "__name__", None)
            )
        if not tool_name:
            tool_name = "tool"

        name_str = str(tool_name)
        if name_str in SENTINEL_TOOL_NAMES:
            continue

        desc = (
            getattr(tool, "desc", None)
            or getattr(tool, "description", None)
            or getattr(tool, "__doc__", None)
        )
        args_schema = getattr(tool, "args", None)

        definitions.append(
            FunctionToolDefinition(
                name=name_str,
                description=str(desc).strip() if desc is not None else None,
                parameters=args_schema,
            )
        )

    return definitions or None
