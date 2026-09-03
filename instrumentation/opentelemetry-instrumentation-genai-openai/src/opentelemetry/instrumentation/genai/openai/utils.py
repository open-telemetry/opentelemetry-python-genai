# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import mimetypes
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

import openai
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
    FilePart,
    FunctionToolDefinition,
    GenericPart,
    InputMessage,
    MessagePart,
    OutputMessage,
    TextPart,
    ToolCallRequestPart,
    ToolCallResponsePart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import image_from_url

_logger = logging.getLogger(__name__)

_OpenAIOmit = getattr(openai, "Omit", None)

SUPPORTED_RAPI_RESPONSE_HEADERS = ("x-ms-served-model",)


def get_served_model(headers: Mapping[str, str] | None) -> str | None:
    """Responses API (RAPI) may include the served model in the
    response headers, which accurately returns the served
    model name for the request."""
    if not isinstance(headers, Mapping):
        return None
    for name, value in headers.items():
        if (
            isinstance(name, str)
            and name.lower() in SUPPORTED_RAPI_RESPONSE_HEADERS
            and isinstance(value, str)
            and value.strip()
        ):
            return str(value)
    return None


def get_property_value(obj, property_name):
    if isinstance(obj, Mapping):
        return obj.get(property_name, None)

    return getattr(obj, property_name, None)


def get_server_address_and_port(
    client_instance,
) -> tuple[str | None, int | None]:
    base_client = getattr(client_instance, "_client", None)
    base_url = getattr(base_client, "base_url", None)
    if not base_url:
        return None, None

    # Use getattr rather than isinstance(base_url, httpx.URL): openai v1/v2
    # uses httpx.URL while v3 uses httpx2.URL; both expose .host and .port.
    address = getattr(base_url, "host", None)
    port = getattr(base_url, "port", None)
    if not address:
        url = urlparse(str(base_url))
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
            invocation.request_choice_count = choice_count

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


def _as_plain_value(item: Any) -> Any:
    """Return a JSON-safe representation of a content part.

    Never raises: pydantic models are dumped in JSON mode, the result is
    round-tripped through ``json`` so only JSON-native values remain, and any
    failure falls back to ``str(item)``.
    """
    try:
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                value = model_dump(mode="json")
            except Exception:  # pylint: disable=broad-exception-caught
                value = model_dump()  # pydantic v1 has no ``mode``
        elif isinstance(item, Mapping):
            value = dict(item)
        else:
            value = item
        return json.loads(json.dumps(value, default=str))
    except Exception:  # pylint: disable=broad-exception-caught
        return str(item)


def _image_url_part(item: Any) -> MessagePart | None:
    """Map an ``image_url`` content part to a ``UriPart`` or ``BlobPart``."""
    image_url = get_property_value(item, "image_url")
    url = (
        image_url
        if isinstance(image_url, str)
        else get_property_value(image_url, "url")
    )
    if not isinstance(url, str) or not url:
        return None
    return image_from_url(url)


def _file_part(item: Any) -> MessagePart | None:
    """Map a ``file`` content part with a ``file_id`` to a ``FilePart``.

    Inline ``file_data`` is intentionally not captured: a single document can
    be megabytes, and it would be base64-inlined into the span attribute.
    """
    file_ref = get_property_value(item, "file")
    file_id = get_property_value(file_ref, "file_id")
    if not isinstance(file_id, str) or not file_id:
        return None
    filename = get_property_value(file_ref, "filename")
    mime_type = None
    if isinstance(filename, str) and filename:
        mime_type = mimetypes.guess_type(filename)[0]
    return FilePart(mime_type=mime_type, modality="document", file_id=file_id)


def _convert_content_part(item: Any) -> MessagePart | None:
    """Map one OpenAI content part to a semconv message part; typed parts
    with no semconv mapping become ``GenericPart`` rather than being dropped.
    Inline media payloads (``input_audio``, ``file_data``) are the exception
    and are never captured."""
    if isinstance(item, str):
        return TextPart(content=item)
    item_type = get_property_value(item, "type")
    if not isinstance(item_type, str):
        return None
    if item_type == "text":
        text = get_property_value(item, "text")
        return TextPart(content=text) if isinstance(text, str) else None
    if item_type == "image_url":
        return _image_url_part(item)
    if item_type == "input_audio":
        # Inline audio is intentionally not captured: a single clip can be
        # megabytes, and it would be base64-inlined into the span attribute.
        return None
    if item_type == "file":
        return _file_part(item)
    return GenericPart(type=item_type, value=_as_plain_value(item))


def _content_to_parts(content: Any) -> list[MessagePart]:
    """Map message ``content`` to message parts.

    A string is a single text part and a bare mapping is a single content
    part. Only sequences are walked as content-part arrays: iterating any
    other iterable (e.g. a generator) would consume the caller's input before
    the SDK sends it, so those are left untouched and not captured.
    Conversion never raises; a part that fails to convert is skipped.
    """
    if isinstance(content, (str, Mapping)):
        content = [content]
    if not isinstance(content, Sequence):
        return []
    parts: list[MessagePart] = []
    for item in content:
        try:
            part = _convert_content_part(item)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Failed to convert content part", exc_info=True)
            continue
        if part is not None:
            parts.append(part)
    return parts


def _prepare_input_messages(messages) -> list[InputMessage]:
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
            chat_message.parts += _content_to_parts(content)

        elif role == "tool":
            tool_call_id = get_property_value(message, "tool_call_id")
            chat_message.parts.append(
                ToolCallResponsePart(id=tool_call_id, response=content)
            )

        else:
            # system, developer, user, fallback
            chat_message.parts += _content_to_parts(content)
    return chat_messages


def extract_tool_calls_new(tool_calls) -> list[ToolCallRequestPart]:
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
            ToolCallRequestPart(
                id=call_id, name=func_name, arguments=arguments
            )
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


def _prepare_output_messages(choices) -> list[OutputMessage]:
    output_messages = []
    for choice in choices:
        if choice.message:
            parts = []
            tool_calls = get_property_value(choice.message, "tool_calls")
            if tool_calls:
                parts += extract_tool_calls_new(tool_calls)
            content = get_property_value(choice.message, "content")
            if _is_text_part(content):
                parts.append(TextPart(content=str(content)))

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
