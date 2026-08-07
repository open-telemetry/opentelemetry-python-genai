# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)

from ._raw_response import ParsableResponse
from .utils import (
    _openai_response_format_to_output_type,
    get_property_value,
    get_served_model,
    get_server_address_and_port,
)

if TYPE_CHECKING:
    from openai.types.responses.response import Response
    from openai.types.responses.response_usage import ResponseUsage

    from opentelemetry.util.genai.types import (
        Error,
        InputMessage,
        OutputMessage,
        TextPart,
        ToolDefinition,
    )

try:
    from openai.types.responses.response import Response
    from openai.types.responses.response_function_tool_call import (
        ResponseFunctionToolCall,
    )
    from openai.types.responses.response_output_message import (
        ResponseOutputMessage,
    )
    from openai.types.responses.response_output_refusal import (
        ResponseOutputRefusal,
    )
    from openai.types.responses.response_output_text import ResponseOutputText
    from openai.types.responses.response_reasoning_item import (
        ResponseReasoningItem,
    )
    from openai.types.responses.response_usage import ResponseUsage
except ImportError:
    Response = None
    ResponseFunctionToolCall = None
    ResponseOutputMessage = None
    ResponseOutputRefusal = None
    ResponseOutputText = None
    ResponseReasoningItem = None
    ResponseUsage = None

try:
    from opentelemetry.util.genai.types import (
        Error,
        FunctionToolDefinition,
        GenericToolDefinition,
        InputMessage,
        OutputMessage,
        ReasoningPart,
        TextPart,
    )
    from opentelemetry.util.genai.types import (
        ToolCallRequestPart as ToolCall,
    )
except ImportError:
    Error = None
    FunctionToolDefinition = None
    GenericToolDefinition = None
    InputMessage = None
    OutputMessage = None
    ReasoningPart = None
    TextPart = None
    ToolCall = None


@dataclass
class ResponseRequestParams:
    model: str | None = None
    instructions: str | None = None
    input: str | Sequence[object] | None = None
    max_output_tokens: int | None = None
    service_tier: str | None = None
    temperature: float | None = None
    output_type: str | None = None
    top_p: float | None = None


@dataclass
class UsageTokens:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


def _get_field(value: object, field_name: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _get_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return value
    return ()


def _get_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _get_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _extract_output_type_from_value(text_config: object) -> str | None:
    format_config = _get_field(text_config, "format")
    if format_config is None:
        return None

    format_type = _get_field(format_config, "type")
    if isinstance(format_type, str):
        return _openai_response_format_to_output_type(format_type)
    return None


def extract_params(
    *,
    model: str | None = None,
    instructions: str | None = None,
    input_items: str | Sequence[object] | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = None,
    temperature: float | None = None,
    text: object | None = None,
    top_p: float | None = None,
    **_kwargs: object,
) -> ResponseRequestParams:
    if input_items is None and "input" in _kwargs:
        input_items = _kwargs["input"]

    return ResponseRequestParams(
        model=model if isinstance(model, str) else None,
        instructions=instructions if isinstance(instructions, str) else None,
        input=(
            input_items
            if isinstance(input_items, str)
            or (
                isinstance(input_items, Sequence)
                and not isinstance(input_items, (str, bytes, bytearray))
            )
            else None
        ),
        max_output_tokens=_get_int(max_output_tokens),
        service_tier=(
            service_tier
            if isinstance(service_tier, str) and service_tier != "auto"
            else None
        ),
        temperature=_get_float(temperature),
        output_type=_extract_output_type_from_value(text),
        top_p=_get_float(top_p),
    )


def get_system_instruction(instructions: str | None) -> list[TextPart]:
    if TextPart is None or instructions is None:
        return []
    return [TextPart(content=instructions)]


def get_input_messages(
    input_value: str | Sequence[object] | None,
) -> list[InputMessage]:
    if InputMessage is None or TextPart is None:
        return []

    if isinstance(input_value, str):
        return [
            InputMessage(role="user", parts=[TextPart(content=input_value)])
        ]

    messages: list[InputMessage] = []
    for item in _get_sequence(input_value):
        role = _get_field(item, "role")
        if not isinstance(role, str):
            continue

        content = _get_field(item, "content")
        if isinstance(content, str):
            messages.append(
                InputMessage(role=role, parts=[TextPart(content=content)])
            )
            continue

        parts = []
        for part in _get_sequence(content):
            text = _get_field(part, "text")
            if isinstance(text, str):
                parts.append(TextPart(content=text))
        if parts:
            messages.append(InputMessage(role=role, parts=parts))

    return messages


def _extract_output_parts(content_blocks: Sequence[object]) -> list[TextPart]:
    if (
        TextPart is None
        or ResponseOutputText is None
        or ResponseOutputRefusal is None
    ):
        return []

    parts: list[TextPart] = []
    for block in content_blocks:
        if isinstance(block, ResponseOutputText):
            parts.append(TextPart(content=block.text))
        elif isinstance(block, ResponseOutputRefusal):
            parts.append(TextPart(content=block.refusal))
    return parts


def _parse_tool_call_arguments(arguments: str | None) -> object:
    if arguments is None:
        return None

    try:
        return json.loads(arguments)
    except (TypeError, ValueError):
        return arguments


def _extract_reasoning_parts(
    item: ResponseReasoningItem,
) -> list[ReasoningPart]:
    if ReasoningPart is None:
        return []

    parts: list[ReasoningPart] = []
    for block in item.summary:
        if isinstance(block.text, str):
            parts.append(ReasoningPart(content=block.text))
    for block in item.content or []:
        if getattr(block, "type", None) == "reasoning_text" and isinstance(
            getattr(block, "text", None), str
        ):
            parts.append(ReasoningPart(content=block.text))
    return parts


# `incomplete_details.reason` values that map onto a cross-provider finish
# reason; an unrecognized reason is reported as-is.
_INCOMPLETE_REASON_TO_FINISH_REASON = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


def _finish_reason_from_status(
    status: str | None,
    incomplete_reason: str | None = None,
) -> str | None:
    if status == "completed":
        return "stop"
    if status in {"failed", "cancelled"}:
        return "error"
    if status == "incomplete":
        if incomplete_reason is None:
            return status
        return _INCOMPLETE_REASON_TO_FINISH_REASON.get(
            incomplete_reason, incomplete_reason
        )
    return None


def get_tool_definitions_from_response(
    response: Response | None,
) -> list[ToolDefinition] | None:
    """Return the tool definitions carried on a fetched response.

    Responses API tools are flat -- a function tool holds ``name``,
    ``description`` and ``parameters`` directly, unlike the Chat Completions
    shape that nests them under ``function``. Built-in tools (``web_search``,
    ``file_search``, ...) are identified by ``type`` alone and carry no name,
    so they are reported as generic definitions keyed by their type.
    """
    if (
        Response is None
        or not isinstance(response, Response)
        or FunctionToolDefinition is None
        or GenericToolDefinition is None
    ):
        return None

    tools = response.tools
    if not tools:
        return None

    definitions: list[ToolDefinition] = []
    for tool in tools:
        tool_type = get_property_value(tool, "type")
        if not isinstance(tool_type, str):
            continue
        name = get_property_value(tool, "name")
        if tool_type == "function":
            definitions.append(
                FunctionToolDefinition(
                    name=name if isinstance(name, str) else "",
                    description=get_property_value(tool, "description"),
                    parameters=get_property_value(tool, "parameters"),
                )
            )
        else:
            definitions.append(
                GenericToolDefinition(
                    name=name if isinstance(name, str) else tool_type,
                    type=tool_type,
                )
            )
    return definitions or None


def _response_types_available() -> bool:
    return (
        Response is not None
        and ResponseOutputMessage is not None
        and ResponseFunctionToolCall is not None
        and ResponseReasoningItem is not None
    )


def get_output_messages_from_response(
    response: Response | None,
) -> list[OutputMessage]:
    if (
        not _response_types_available()
        or not isinstance(response, Response)
        or OutputMessage is None
        or TextPart is None
    ):
        return []

    messages: list[OutputMessage] = []
    for item in response.output:
        if isinstance(item, ResponseOutputMessage):
            finish_reason = _finish_reason_from_status(item.status)
            if finish_reason is None:
                continue

            messages.append(
                OutputMessage(
                    role=item.role,
                    parts=_extract_output_parts(item.content),
                    finish_reason=finish_reason,
                )
            )
            continue

        if isinstance(item, ResponseFunctionToolCall):
            if ToolCall is None or item.status not in {
                "completed",
                "incomplete",
            }:
                continue

            messages.append(
                OutputMessage(
                    role="assistant",
                    parts=[
                        ToolCall(
                            id=item.call_id if item.call_id else item.id,
                            name=item.name,
                            arguments=_parse_tool_call_arguments(
                                item.arguments
                            ),
                        )
                    ],
                    finish_reason="tool_calls",
                )
            )
            continue

        if isinstance(item, ResponseReasoningItem):
            finish_reason = _finish_reason_from_status(item.status)
            if finish_reason is None:
                continue

            parts = _extract_reasoning_parts(item)
            if parts:
                messages.append(
                    OutputMessage(
                        role="assistant",
                        parts=parts,
                        finish_reason=finish_reason,
                    )
                )

    return messages


def extract_finish_reasons(response: Response | None) -> list[str]:
    if (
        Response is None
        or ResponseOutputMessage is None
        or ResponseFunctionToolCall is None
        or not isinstance(response, Response)
    ):
        return []

    incomplete_details = response.incomplete_details
    response_finish_reason = _finish_reason_from_status(
        response.status,
        incomplete_details.reason if incomplete_details is not None else None,
    )
    if response.status in {"failed", "cancelled", "incomplete"}:
        return [response_finish_reason] if response_finish_reason else []
    if response.status in {"queued", "in_progress"}:
        return []

    finish_reasons: list[str] = []
    for item in response.output:
        if isinstance(item, ResponseFunctionToolCall) and item.status in {
            "completed",
            "incomplete",
        }:
            finish_reasons.append("tool_calls")
            continue

        if not isinstance(item, ResponseOutputMessage):
            continue
        finish_reason = _finish_reason_from_status(item.status)
        if finish_reason is not None:
            finish_reasons.append(finish_reason)
    finish_reasons = list(dict.fromkeys(finish_reasons))
    if finish_reasons:
        return finish_reasons
    return [response_finish_reason] if response_finish_reason else []


def get_response_error(
    response: object,
    request_kwargs: dict[str, object] | None = None,
) -> Error | None:
    """Return an ``Error`` when the response failed, else ``None``.

    A failed response carries a ``ResponseError`` (``code`` + ``message``).
    Incomplete responses (``incomplete_details``) are *not* errors — they
    surface as a finish reason instead.
    """
    response = _parse_raw_response(response, request_kwargs)

    if Response is None or Error is None or not isinstance(response, Response):
        return None
    error = response.error
    if error is None:
        return None
    return Error(type=error.code, message=error.message)


def get_inference_creation_kwargs(
    params: ResponseRequestParams,
    client_instance: object,
) -> dict[str, object]:
    address, port = get_server_address_and_port(client_instance)

    creation_kwargs: dict[str, object] = {
        "provider": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    }
    if params.model is not None:
        creation_kwargs["request_model"] = params.model
    if address is not None:
        creation_kwargs["server_address"] = address
    if port is not None:
        creation_kwargs["server_port"] = port
    return creation_kwargs


def get_fetch_response_creation_kwargs(
    response_id: str,
    client_instance: object,
) -> dict[str, object]:
    """Return ``handler.fetch_response()`` kwargs for a ``responses.retrieve`` call."""
    address, port = get_server_address_and_port(client_instance)

    creation_kwargs: dict[str, object] = {
        "provider": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
        "response_id": response_id,
    }
    if address is not None:
        creation_kwargs["server_address"] = address
    if port is not None:
        creation_kwargs["server_port"] = port
    return creation_kwargs


def apply_request_attributes(
    invocation,
    params: ResponseRequestParams,
    capture_content: bool,
) -> None:
    invocation.attributes[OpenAIAttributes.OPENAI_API_TYPE] = (
        OpenAIAttributes.OpenaiApiTypeValues.RESPONSES.value
    )
    invocation.temperature = params.temperature
    invocation.top_p = params.top_p
    invocation.max_tokens = params.max_output_tokens

    if params.service_tier is not None:
        invocation.attributes[OpenAIAttributes.OPENAI_REQUEST_SERVICE_TIER] = (
            params.service_tier
        )

    if params.output_type is not None:
        invocation.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] = (
            params.output_type
        )

    if capture_content:
        invocation.system_instruction = get_system_instruction(
            params.instructions
        )
        invocation.input_messages = get_input_messages(params.input)


def extract_usage_tokens(usage: ResponseUsage | None) -> UsageTokens:
    if (
        ResponseUsage is None
        or usage is None
        or not isinstance(usage, ResponseUsage)
    ):
        return UsageTokens()

    details = usage.input_tokens_details
    return UsageTokens(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        # `cache_creation_input_tokens` is not present on every SDK version's
        # input token details model, so keep this attribute access guarded for
        # compatibility across the supported OpenAI range.
        cache_creation_input_tokens=(
            details.cache_creation_input_tokens
            if details is not None
            and hasattr(details, "cache_creation_input_tokens")
            else None
        ),
        cache_read_input_tokens=(
            details.cached_tokens if details is not None else None
        ),
    )


_RAW_RESPONSE_HEADER = "x-stainless-raw-response"


def is_streamed_raw_response(
    request_kwargs: dict[str, object] | None,
) -> bool:
    if request_kwargs is None:
        return False
    extra_headers = request_kwargs.get("extra_headers")
    if not isinstance(extra_headers, Mapping):
        return False
    return any(
        key.lower() == _RAW_RESPONSE_HEADER and value == "stream"
        for key, value in extra_headers.items()
    )


def _parse_raw_response(
    response: object,
    request_kwargs: dict[str, object] | None,
) -> object:
    """Return the payload of a non-streaming ``with_raw_response`` result."""
    if is_streamed_raw_response(request_kwargs) or not isinstance(
        response, ParsableResponse
    ):
        return response
    return response.parse()


def set_invocation_response_attributes(
    invocation,
    response: object,
    capture_content: bool,
    request_kwargs: dict[str, object] | None = None,
) -> None:
    served_model = get_served_model(getattr(response, "headers", None))
    response = _parse_raw_response(response, request_kwargs)

    if Response is None or not isinstance(response, Response):
        return
    if served_model:
        invocation.response_model_name = served_model
    else:
        invocation.response_model_name = response.model
    invocation.response_id = response.id

    if response.service_tier is not None:
        invocation.attributes[
            OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        ] = response.service_tier

    tokens = extract_usage_tokens(response.usage)
    invocation.input_tokens = tokens.input_tokens
    invocation.output_tokens = tokens.output_tokens
    invocation.cache_creation_input_tokens = tokens.cache_creation_input_tokens
    invocation.cache_read_input_tokens = tokens.cache_read_input_tokens

    finish_reasons = extract_finish_reasons(response)
    if finish_reasons:
        invocation.finish_reasons = finish_reasons

    if capture_content:
        output_messages = get_output_messages_from_response(response)
        if output_messages:
            invocation.output_messages = output_messages


def set_fetch_response_attributes(
    invocation,
    response: object,
    capture_content: bool,
    request_kwargs: dict[str, object] | None = None,
) -> None:
    """Record a fetched response on a ``fetch_response`` invocation.

    Token usage is deliberately not recorded: the fetch performs no inference
    and the counts on the fetched response belong to the original generation.
    The original input messages are not part of the fetched response either, so
    only the system instructions and output messages it carries are captured.
    """
    served_model = get_served_model(getattr(response, "headers", None))
    response = _parse_raw_response(response, request_kwargs)

    if Response is None or not isinstance(response, Response):
        return

    invocation.response_model_name = served_model or response.model
    invocation.response_status = response.status
    invocation.finish_reasons = extract_finish_reasons(response) or None

    # `service_tier` is absent from the Response model on some supported SDK
    # versions, so keep this attribute access guarded.
    service_tier = getattr(response, "service_tier", None)
    if service_tier is not None:
        invocation.attributes[
            OpenAIAttributes.OPENAI_RESPONSE_SERVICE_TIER
        ] = service_tier

    if capture_content:
        invocation.system_instruction = get_system_instruction(
            response.instructions
            if isinstance(response.instructions, str)
            else None
        )
        invocation.output_messages = get_output_messages_from_response(
            response
        )
        invocation.tool_definitions = get_tool_definitions_from_response(
            response
        )
