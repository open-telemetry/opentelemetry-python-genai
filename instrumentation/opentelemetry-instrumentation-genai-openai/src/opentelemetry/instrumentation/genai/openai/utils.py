# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any, Iterable, List, Mapping
from urllib.parse import urlparse

import openai
from httpx import URL
from openai import NotGiven

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

_OpenAIOmit = getattr(openai, "Omit", None)


def get_property_value(obj, property_name):
    if isinstance(obj, dict):
        return obj.get(property_name, None)

    return getattr(obj, property_name, None)


def get_server_address_and_port(
    client_instance,
) -> tuple[str | None, int | None]:
    base_client = getattr(client_instance, "_client", None)
    base_url = getattr(base_client, "base_url", None)
    if not base_url:
        return None, None
    address = None
    port = None
    if isinstance(base_url, URL):
        address = base_url.host
        port = base_url.port
    elif isinstance(base_url, str):
        url = urlparse(base_url)
        address = url.hostname
        port = url.port

    if port == 443:
        port = None

    return address, port


def is_streaming(kwargs):
    return non_numerical_value_is_set(kwargs.get("stream"))


def non_numerical_value_is_set(value: bool | str | NotGiven | None):
    return bool(value) and value_is_set(value)


def value_is_set(value):
    if _OpenAIOmit is not None and isinstance(value, _OpenAIOmit):
        return False
    return value is not None and not isinstance(value, NotGiven)


def _openai_response_format_to_output_type(response_format_type: str) -> str:
    if response_format_type in ("json_object", "json_schema"):
        return GenAIAttributes.GenAiOutputTypeValues.JSON.value
    return response_format_type


def create_chat_invocation(
    handler: TelemetryHandler,
    kwargs,
    client_instance,
    capture_content: bool,
) -> InferenceInvocation:
    # pylint: disable=too-many-branches

    address, port = get_server_address_and_port(client_instance)
    invocation = handler.inference(
        GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        request_model=kwargs.get("model", ""),
        server_address=address if address else None,
        server_port=port if port else None,
    )
    invocation.temperature = get_value(kwargs.get("temperature"))
    invocation.top_p = get_value(kwargs.get("p") or kwargs.get("top_p"))
    invocation.max_tokens = get_value(kwargs.get("max_tokens"))
    invocation.presence_penalty = get_value(kwargs.get("presence_penalty"))
    invocation.frequency_penalty = get_value(kwargs.get("frequency_penalty"))
    invocation.seed = get_value(kwargs.get("seed"))
    if (stop_sequences := get_value(kwargs.get("stop"))) is not None:
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        invocation.stop_sequences = stop_sequences

    if (choice_count := get_value(kwargs.get("n"))) is not None:
        # Only add non default, meaningful values
        if isinstance(choice_count, int) and choice_count != 1:
            invocation.attributes[
                GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT
            ] = choice_count

    if (
        response_format := get_value(kwargs.get("response_format"))
    ) is not None:
        # response_format may be string, object with a string in the `type` key,
        # or a type (e.g. Pydantic model class used with parse())
        if isinstance(response_format, type):
            invocation.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                GenAIAttributes.GenAiOutputTypeValues.JSON.value
            )
        elif isinstance(response_format, Mapping):
            if (
                response_format_type := get_value(response_format.get("type"))
            ) is not None:
                invocation.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                    _openai_response_format_to_output_type(
                        response_format_type
                    )
                )
        elif isinstance(response_format, str):
            invocation.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
                _openai_response_format_to_output_type(response_format)
            )

    # service_tier can be passed directly or in extra_body (in SDK 1.26.0 it's via extra_body)
    service_tier = get_value(kwargs.get("service_tier"))
    if service_tier is None:
        extra_body = get_value(kwargs.get("extra_body"))
        if isinstance(extra_body, Mapping):
            service_tier = get_value(extra_body.get("service_tier"))
    if service_tier is not None and service_tier != "auto":
        invocation.attributes[OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER] = (
            service_tier
        )

    if capture_content:  # optimization
        invocation.input_messages = _prepare_input_messages(
            kwargs.get("messages", [])
        )
        invocation.tool_definitions = _prepare_tool_definitions(
            kwargs.get("tools")
        )
    return invocation


def get_value(v: Any):
    if value_is_set(v):
        return v
    return None


def _is_text_part(content: Any) -> bool:
    return isinstance(content, str) or (
        isinstance(content, Iterable)
        and all(isinstance(part, str) for part in content)
    )


def _prepare_input_messages(messages) -> List[InputMessage]:
    chat_messages = []
    for message in messages:
        role = get_property_value(message, "role")
        chat_message = InputMessage(role=str(role), parts=[])
        chat_messages.append(chat_message)

        content = get_property_value(message, "content")

        if role == "assistant":
            tool_calls = get_property_value(message, "tool_calls")
            if tool_calls:
                chat_message.parts += extract_tool_calls_new(tool_calls)
            if _is_text_part(content):
                chat_message.parts.append(Text(content=str(content)))

        elif role == "tool":
            tool_call_id = get_property_value(message, "tool_call_id")
            chat_message.parts.append(
                ToolCallResponse(id=tool_call_id, response=content)
            )

        else:
            # system, developer, user, fallback
            if _is_text_part(content):
                chat_message.parts.append(Text(content=str(content)))
    return chat_messages


def extract_tool_calls_new(tool_calls) -> list[ToolCallRequest]:
    parts = []
    for tool_call in tool_calls:
        call_id = get_property_value(tool_call, "id")

        func_name = ""
        arguments = None
        func = get_property_value(tool_call, "function")
        if func:
            func_name = get_property_value(func, "name") or ""
            arguments_str = get_property_value(func, "arguments")
            if arguments_str:
                try:
                    arguments = json.loads(arguments_str)
                except json.JSONDecodeError:
                    arguments = arguments_str

        # TODO: support custom
        parts.append(
            ToolCallRequest(id=call_id, name=func_name, arguments=arguments)
        )
    return parts


def _prepare_tool_definitions(tools) -> list[ToolDefinition] | None:
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = get_property_value(tool, "type")
        if tool_type == "function":
            func = get_property_value(tool, "function")
            if func:
                definitions.append(
                    FunctionToolDefinition(
                        name=get_property_value(func, "name") or "",
                        description=get_property_value(func, "description"),
                        parameters=get_property_value(func, "parameters"),
                    )
                )
    return definitions


def _prepare_output_messages(choices) -> List[OutputMessage]:
    output_messages = []
    for choice in choices:
        if choice.message:
            parts = []
            tool_calls = get_property_value(choice.message, "tool_calls")
            if tool_calls:
                parts += extract_tool_calls_new(tool_calls)
            content = get_property_value(choice.message, "content")
            if _is_text_part(content):
                parts.append(Text(content=str(content)))

            message = OutputMessage(
                finish_reason=choice.finish_reason or "error",
                role=(
                    choice.message.role
                    if choice.message and choice.message.role
                    else ""
                ),
                parts=parts,
            )
            output_messages.append(message)

    return output_messages
