# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Utilities for extracting telemetry from Portkey AI requests and responses."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import InferenceInvocation
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

if TYPE_CHECKING:
    from portkey_ai import AsyncPortkey, Portkey

# Mapping from Portkey ``provider`` values to the well-known
# ``gen_ai.provider.name`` values defined by the GenAI semantic conventions.
_PROVIDER_NAME_OVERRIDES: dict[str, str] = {
    "azure-openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "azure-ai-inference": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_INFERENCE.value,
    "bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "aws-bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "amazon-bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "vertex-ai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "vertexai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google-vertex-ai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "google": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "gemini": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "google-generativeai": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "mistral": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "mistral-ai": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "mistralai": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "perplexity": GenAIAttributes.GenAiProviderNameValues.PERPLEXITY.value,
    "perplexity-ai": GenAIAttributes.GenAiProviderNameValues.PERPLEXITY.value,
    "x-ai": GenAIAttributes.GenAiProviderNameValues.X_AI.value,
    "xai": GenAIAttributes.GenAiProviderNameValues.X_AI.value,
    "watsonx": GenAIAttributes.GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "ibm-watsonx": GenAIAttributes.GenAiProviderNameValues.IBM_WATSONX_AI.value,
    "deepseek": GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,
    "groq": GenAIAttributes.GenAiProviderNameValues.GROQ.value,
}


def get_property_value(obj: Any, property_name: str) -> Any:
    """Extract a property value from either a dict or an object attribute."""
    if isinstance(obj, dict):
        return cast(dict[str, Any], obj).get(property_name)
    return getattr(obj, property_name, None)


def value_is_set(value: Any) -> bool:
    """Check if a value is meaningful (not None, Omit, or NotGiven)."""
    if value is None:
        return False
    type_name = type(value).__name__
    if type_name in ("Omit", "NotGiven"):
        return False
    return True


def get_value(value: Any) -> Any:
    """Return the value if set, else None."""
    return value if value_is_set(value) else None


def is_streaming(kwargs: dict[str, Any]) -> bool:
    """Check whether the request is configured for streaming."""
    stream = kwargs.get("stream")
    if not value_is_set(stream):
        return False
    return bool(stream)


def get_provider(client_instance: Portkey | AsyncPortkey) -> str:
    """Derive the provider name from the Portkey client instance."""
    provider = getattr(client_instance, "provider", None)
    if provider and isinstance(provider, str):
        provider_lower = provider.lower()
        return _PROVIDER_NAME_OVERRIDES.get(provider_lower, provider_lower)
    return "portkey"


def get_server_address_and_port(
    client_instance: Portkey | AsyncPortkey,
) -> tuple[str | None, int | None]:
    """Extract the server host and port from client base_url."""
    base_url = getattr(client_instance, "base_url", None)
    if not base_url:
        return None, None

    url = urlparse(str(base_url))
    address = url.hostname
    port = url.port
    if port == 443:
        port = None
    return address, port


def _extract_tool_calls(tool_calls: Iterable[Any]) -> list[ToolCallRequest]:
    parts: list[ToolCallRequest] = []
    for tool_call in tool_calls:
        call_id = get_property_value(tool_call, "id")
        func_name = ""
        arguments: Any = None
        func = get_property_value(tool_call, "function")
        if func is not None:
            func_name = str(get_property_value(func, "name") or "")
            args_raw = get_property_value(func, "arguments")
            if isinstance(args_raw, str):
                try:
                    arguments = json.loads(args_raw)
                except Exception:
                    arguments = args_raw
            elif args_raw is not None:
                arguments = args_raw
        parts.append(
            ToolCallRequest(
                id=str(call_id) if call_id is not None else None,
                name=func_name,
                arguments=arguments,
            )
        )
    return parts


def _prepare_input_messages(messages: Iterable[Any]) -> list[InputMessage]:
    chat_messages: list[InputMessage] = []
    for message in messages:
        role = get_property_value(message, "role")
        if role is None:
            continue
        parts: list[MessagePart] = []
        content = get_property_value(message, "content")
        tool_calls = get_property_value(message, "tool_calls")
        tool_call_id = get_property_value(message, "tool_call_id")

        if tool_calls is not None and isinstance(tool_calls, Iterable):
            parts.extend(_extract_tool_calls(cast(Iterable[Any], tool_calls)))
        if tool_call_id is not None:
            parts.append(
                ToolCallResponse(id=str(tool_call_id), response=content)
            )
        elif content:
            if isinstance(content, str):
                parts.append(Text(content=content))
            elif isinstance(content, Mapping):
                content_dict = cast(Mapping[str, Any], content)
                content_type = content_dict.get("type")
                if content_type == "text" and content_dict.get("text"):
                    parts.append(Text(content=str(content_dict["text"])))
            elif isinstance(content, Iterable):
                for item in cast(Iterable[Any], content):
                    if isinstance(item, str):
                        parts.append(Text(content=item))
                    elif isinstance(item, Mapping):
                        item_dict = cast(Mapping[str, Any], item)
                        item_type = item_dict.get("type")
                        if item_type == "text" and item_dict.get("text"):
                            parts.append(Text(content=str(item_dict["text"])))
        chat_messages.append(InputMessage(role=str(role), parts=parts))
    return chat_messages


def _prepare_tool_definitions(
    tools: Iterable[Any] | None,
) -> list[ToolDefinition] | None:
    if not tools:
        return None
    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = get_property_value(tool, "type")
        if tool_type == "function":
            func = get_property_value(tool, "function")
            if func is not None:
                definitions.append(
                    FunctionToolDefinition(
                        name=str(get_property_value(func, "name") or ""),
                        description=get_property_value(func, "description"),
                        parameters=get_property_value(func, "parameters"),
                    )
                )
    return definitions or None


def _prepare_output_messages(
    choices: Iterable[Any] | None,
) -> list[OutputMessage]:
    if not choices:
        return []
    output_messages: list[OutputMessage] = []
    for choice in choices:
        finish_reason = get_property_value(choice, "finish_reason") or "stop"
        parts: list[MessagePart] = []
        message = get_property_value(choice, "message")
        role = "assistant"
        if message is not None:
            msg_role = get_property_value(message, "role")
            if msg_role:
                role = str(msg_role)
            content = get_property_value(message, "content")
            if content:
                parts.append(Text(content=str(content)))
            tool_calls = get_property_value(message, "tool_calls")
            if tool_calls is not None and isinstance(tool_calls, Iterable):
                parts.extend(
                    _extract_tool_calls(cast(Iterable[Any], tool_calls))
                )
        else:
            text = get_property_value(choice, "text")
            if text:
                parts.append(Text(content=str(text)))
        output_messages.append(
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=str(finish_reason),
            )
        )
    return output_messages


def _apply_request_parameters(
    invocation: InferenceInvocation,
    kwargs: dict[str, Any],
    capture_content: bool,
) -> None:
    if (temp := get_value(kwargs.get("temperature"))) is not None:
        invocation.temperature = float(temp)
    if (top_p := get_value(kwargs.get("top_p"))) is not None:
        invocation.top_p = float(top_p)
    elif (p := get_value(kwargs.get("p"))) is not None:
        invocation.top_p = float(p)
    if (top_k := get_value(kwargs.get("top_k"))) is not None:
        invocation.top_k = float(top_k)
    if (max_tokens := get_value(kwargs.get("max_tokens"))) is not None:
        invocation.max_tokens = int(max_tokens)
    elif (
        max_completion_tokens := get_value(kwargs.get("max_completion_tokens"))
    ) is not None:
        invocation.max_tokens = int(max_completion_tokens)
    if (stop := get_value(kwargs.get("stop"))) is not None:
        if isinstance(stop, str):
            invocation.stop_sequences = [stop]
        elif isinstance(stop, Iterable):
            invocation.stop_sequences = [
                str(s) for s in cast(Iterable[Any], stop)
            ]
    if (
        presence_penalty := get_value(kwargs.get("presence_penalty"))
    ) is not None:
        invocation.presence_penalty = float(presence_penalty)
    if (
        frequency_penalty := get_value(kwargs.get("frequency_penalty"))
    ) is not None:
        invocation.frequency_penalty = float(frequency_penalty)
    if (seed := get_value(kwargs.get("seed"))) is not None:
        invocation.seed = int(seed)
    if (choice_count := get_value(kwargs.get("n"))) is not None:
        if isinstance(choice_count, int) and choice_count != 1:
            invocation.request_choice_count = choice_count

    if (
        response_format := get_value(kwargs.get("response_format"))
    ) is not None:
        if isinstance(response_format, type):
            invocation.output_type = (
                GenAIAttributes.GenAiOutputTypeValues.JSON.value
            )
        elif isinstance(response_format, Mapping):
            rf_map = cast(Mapping[str, Any], response_format)
            rf_type = get_value(rf_map.get("type"))
            if rf_type in ("json_object", "json_schema"):
                invocation.output_type = (
                    GenAIAttributes.GenAiOutputTypeValues.JSON.value
                )
            elif isinstance(rf_type, str):
                invocation.output_type = rf_type
        elif isinstance(response_format, str):
            if response_format in ("json_object", "json_schema"):
                invocation.output_type = (
                    GenAIAttributes.GenAiOutputTypeValues.JSON.value
                )
            else:
                invocation.output_type = response_format

    tools = kwargs.get("tools")
    if tools is not None and isinstance(tools, Iterable):
        invocation.tool_definitions = _prepare_tool_definitions(
            cast(Iterable[Any], tools)
        )

    if capture_content:
        messages = kwargs.get("messages")
        if messages is not None and isinstance(messages, Iterable):
            invocation.input_messages = _prepare_input_messages(
                cast(Iterable[Any], messages)
            )


def create_inference_invocation(
    handler: TelemetryHandler,
    instance: Any,
    kwargs: dict[str, Any],
    capture_content: bool,
    *,
    is_prompt: bool = False,
) -> InferenceInvocation:
    """Create an InferenceInvocation for a chat or prompt completions request."""
    client = getattr(instance, "_client", instance)
    provider = get_provider(client)
    address, port = get_server_address_and_port(client)

    model = get_value(kwargs.get("model"))
    if model is None and not is_prompt:
        model = "portkey-default"

    invocation = handler.inference(
        provider=provider,
        request_model=str(model) if model is not None else None,
        server_address=address,
        server_port=port,
    )
    if is_prompt:
        if prompt_id := get_value(kwargs.get("prompt_id")):
            invocation.attributes[GenAIAttributes.GEN_AI_PROMPT_NAME] = str(
                prompt_id
            )

    _apply_request_parameters(invocation, kwargs, capture_content)
    return invocation


def set_response_properties(
    invocation: InferenceInvocation,
    result: Any,
    capture_content: bool,
) -> None:
    """Populate invocation with response attributes from a completion object."""
    if result is None:
        return

    if (resp_id := get_property_value(result, "id")) is not None:
        invocation.response_id = str(resp_id)

    if (model := get_property_value(result, "model")) is not None:
        invocation.response_model_name = str(model)

    choices = get_property_value(result, "choices")
    if choices is not None and isinstance(choices, Iterable):
        invocation.finish_reasons = [
            str(fr)
            for choice in cast(Iterable[Any], choices)
            if (fr := get_property_value(choice, "finish_reason")) is not None
        ]
        if capture_content:
            invocation.output_messages = _prepare_output_messages(
                cast(Iterable[Any], choices)
            )

    usage = get_property_value(result, "usage")
    if usage is not None:
        prompt_tokens = get_property_value(usage, "prompt_tokens")
        if prompt_tokens is not None:
            invocation.input_tokens = int(prompt_tokens)
        completion_tokens = get_property_value(usage, "completion_tokens")
        if completion_tokens is not None:
            invocation.output_tokens = int(completion_tokens)
