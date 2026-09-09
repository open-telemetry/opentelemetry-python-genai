# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Get/extract helpers for Anthropic Messages instrumentation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from urllib.parse import urlparse

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib
from anthropic.types import MessageDeltaUsage

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    GenericToolDefinition,
    InputMessage,
    OutputMessage,
    SystemInstructionPart,
    TextPart,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

from .utils import (
    convert_content_to_parts,
    normalize_finish_reason,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anthropic.resources.messages import AsyncMessages, Messages
    from anthropic.types import (
        Message,
        MessageParam,
        MetadataParam,
        TextBlockParam,
        ThinkingConfigParam,
        ToolChoiceParam,
        ToolUnionParam,
        Usage,
    )


@dataclass
class MessageRequestParams:
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    stop_sequences: Sequence[str] | None = None
    stream: bool | None = None
    messages: Iterable[MessageParam] | None = None
    system: str | Iterable[TextBlockParam] | None = None
    tools: Iterable[ToolUnionParam] | None = None


@dataclass
class UsageTokens:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


def extract_usage_tokens(
    usage: Usage | MessageDeltaUsage | None,
) -> UsageTokens:
    if usage is None:
        return UsageTokens()

    input_tokens = usage.input_tokens
    output_tokens = usage.output_tokens
    cache_creation_input_tokens = usage.cache_creation_input_tokens
    cache_read_input_tokens = usage.cache_read_input_tokens

    if (
        input_tokens is None
        and cache_creation_input_tokens is None
        and cache_read_input_tokens is None
    ):
        total_input_tokens = None
    else:
        total_input_tokens = (
            (input_tokens or 0)
            + (cache_creation_input_tokens or 0)
            + (cache_read_input_tokens or 0)
        )

    return UsageTokens(
        input_tokens=total_input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def get_input_messages(
    messages: Iterable[MessageParam] | None,
) -> list[InputMessage]:
    if messages is None:
        return []
    result: list[InputMessage] = []
    for message in messages:
        role = message["role"]
        parts = convert_content_to_parts(message["content"])
        result.append(InputMessage(role=role, parts=parts))
    return result


def get_system_instruction(
    system: str | Iterable[TextBlockParam] | None,
) -> list[SystemInstructionPart]:
    if system is None:
        return []
    if isinstance(system, str):
        return [TextPart(content=system)] if system else []
    return [
        TextPart(content=block["text"])
        for block in system
        if block.get("text")
    ]


def _tool_field(tool: object, key: str) -> object:
    if isinstance(tool, Mapping):
        return cast("Mapping[str, object]", tool).get(key)
    return getattr(tool, key, None)


def get_tool_definitions(
    tools: Iterable[ToolUnionParam] | None,
) -> list[ToolDefinition] | None:
    """Convert the request's ``tools`` into semconv tool definitions.

    A custom tool carries its JSON schema in ``input_schema`` and maps onto a
    function definition. Server tools are identified by a versioned ``type``
    (``web_search_20250305``, ``bash_20250124``, ...) and toolsets
    (``computer_toolset_20260801``, ...) carry no ``name`` at all, so their type
    stands in for the name.
    """
    if tools is None:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        name = _tool_field(tool, "name")
        tool_type = _tool_field(tool, "type")
        input_schema = _tool_field(tool, "input_schema")
        if input_schema is not None or tool_type in (None, "custom"):
            description = _tool_field(tool, "description")
            definitions.append(
                FunctionToolDefinition(
                    name=name if isinstance(name, str) else "",
                    description=(
                        description if isinstance(description, str) else None
                    ),
                    # The schema requires an object; drop anything else rather
                    # than emit a tool definition that fails validation.
                    parameters=(
                        input_schema
                        if isinstance(input_schema, Mapping)
                        else None
                    ),
                )
            )
        elif isinstance(tool_type, str):
            definitions.append(
                GenericToolDefinition(
                    name=name if isinstance(name, str) else tool_type,
                    type=tool_type,
                )
            )
    return definitions or None


def get_output_messages_from_message(
    message: Message | None,
) -> list[OutputMessage]:
    if message is None:
        return []

    parts = convert_content_to_parts(message.content)
    finish_reason = normalize_finish_reason(message.stop_reason)
    return [
        OutputMessage(
            role=message.role,
            parts=parts,
            finish_reason=finish_reason or "",
        )
    ]


def set_invocation_response_attributes(
    invocation: InferenceInvocation,
    message: Message | None,
    capture_content: bool,
) -> None:
    if message is None:
        return

    invocation.response_model_name = message.model
    invocation.response_id = message.id

    finish_reason = normalize_finish_reason(message.stop_reason)
    if finish_reason:
        invocation.finish_reasons = [finish_reason]

    tokens = extract_usage_tokens(message.usage)
    invocation.input_tokens = tokens.input_tokens
    invocation.output_tokens = tokens.output_tokens
    invocation.cache_creation_input_tokens = tokens.cache_creation_input_tokens
    invocation.cache_read_input_tokens = tokens.cache_read_input_tokens

    if capture_content:
        invocation.output_messages = get_output_messages_from_message(message)


def extract_params(  # pylint: disable=too-many-locals
    *,
    max_tokens: int | None = None,
    messages: Iterable[MessageParam] | None = None,
    model: str | None = None,
    metadata: MetadataParam | None = None,
    service_tier: str | None = None,
    stop_sequences: Sequence[str] | None = None,
    stream: bool | None = None,
    system: str | Iterable[TextBlockParam] | None = None,
    temperature: float | None = None,
    thinking: ThinkingConfigParam | None = None,
    tool_choice: ToolChoiceParam | None = None,
    tools: Iterable[ToolUnionParam] | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra_query: Mapping[str, object] | None = None,
    extra_body: object | None = None,
    timeout: float | _http_lib.Timeout | None = None,
    **_kwargs: object,
) -> MessageRequestParams:
    if isinstance(extra_body, Mapping):
        body = cast(Mapping[object, object], extra_body)
        body_temperature = body.get("temperature")
        if (
            temperature is None
            and isinstance(body_temperature, (int, float))
            and not isinstance(body_temperature, bool)
        ):
            temperature = float(body_temperature)

        body_top_p = body.get("top_p")
        if (
            top_p is None
            and isinstance(body_top_p, (int, float))
            and not isinstance(body_top_p, bool)
        ):
            top_p = float(body_top_p)

        body_top_k = body.get("top_k")
        if (
            top_k is None
            and isinstance(body_top_k, int)
            and not isinstance(body_top_k, bool)
        ):
            top_k = body_top_k

    return MessageRequestParams(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_sequences=stop_sequences,
        stream=stream,
        messages=messages,
        system=system,
        tools=tools,
    )


def get_server_address_and_port(
    client_instance: Messages | AsyncMessages,
) -> tuple[str | None, int | None]:
    base_client = getattr(client_instance, "_client", None)
    base_url = getattr(base_client, "base_url", None)
    if not base_url:
        return None, None

    server_address = getattr(base_url, "host", None)
    server_port = getattr(base_url, "port", None)

    if server_address is None:
        parsed = urlparse(str(base_url))
        server_address = parsed.hostname
        server_port = parsed.port

    if server_port in (80, 443):
        server_port = None

    return server_address, server_port


def get_llm_request_attributes(
    params: MessageRequestParams, client_instance: Messages | AsyncMessages
) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue | None] = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAIAttributes.GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_PROVIDER_NAME: (
            GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
        ),
        GenAIAttributes.GEN_AI_REQUEST_MODEL: params.model,
        GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: params.max_tokens,
        GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: params.temperature,
        GenAIAttributes.GEN_AI_REQUEST_TOP_P: params.top_p,
        GenAIAttributes.GEN_AI_REQUEST_TOP_K: params.top_k,
        GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: params.stop_sequences,
    }
    address, port = get_server_address_and_port(client_instance)
    if address is not None:
        attributes[ServerAttributes.SERVER_ADDRESS] = address
    if port is not None:
        attributes[ServerAttributes.SERVER_PORT] = port
    return {k: v for k, v in attributes.items() if v is not None}
