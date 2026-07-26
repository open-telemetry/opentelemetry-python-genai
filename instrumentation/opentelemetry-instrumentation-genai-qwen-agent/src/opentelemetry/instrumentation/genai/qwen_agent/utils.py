# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversion helpers between qwen-agent message types and
``opentelemetry.util.genai`` semantic convention types."""

from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)

_logger = logging.getLogger(__name__)

# DashScope is not part of the gen_ai.provider.name well-known values yet;
# the attribute is an open enum so a custom string is allowed.
_PROVIDER_DASHSCOPE = "dashscope"

# Map qwen-agent model_type to a gen_ai.provider.name value.
_MODEL_TYPE_PROVIDER_MAP = {
    "qwen_dashscope": _PROVIDER_DASHSCOPE,
    "qwenvl_dashscope": _PROVIDER_DASHSCOPE,
    "qwenaudio_dashscope": _PROVIDER_DASHSCOPE,
    "qwenvlo_dashscope": _PROVIDER_DASHSCOPE,
    "oai": GenAI.GenAiProviderNameValues.OPENAI.value,
    "azure": GenAI.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "qwenvl_oai": GenAI.GenAiProviderNameValues.OPENAI.value,
    "qwenomni_oai": GenAI.GenAiProviderNameValues.OPENAI.value,
}


def _get_provider_name(llm_instance: Any) -> str:
    """Infer the gen_ai.provider.name value for a qwen-agent LLM instance."""
    model_type = getattr(llm_instance, "model_type", "")
    if model_type in _MODEL_TYPE_PROVIDER_MAP:
        return _MODEL_TYPE_PROVIDER_MAP[model_type]

    class_name = type(llm_instance).__name__.lower()
    if "dashscope" in class_name:
        return _PROVIDER_DASHSCOPE
    if "openai" in class_name or "oai" in class_name:
        return GenAI.GenAiProviderNameValues.OPENAI.value
    if "azure" in class_name:
        return GenAI.GenAiProviderNameValues.AZURE_AI_OPENAI.value

    return _PROVIDER_DASHSCOPE


def _field_value(value: Any, *names: str) -> Any:
    """Read the first present field from a dict or SDK object."""
    if value is None:
        return None

    for name in names:
        if isinstance(value, dict):
            if name in value:
                return value[name]
            continue

        attr_value = getattr(value, name, None)
        if attr_value is not None:
            return attr_value

        get_method = getattr(value, "get", None)
        if callable(get_method):
            try:
                got_value = get_method(name)
            except Exception:  # pylint: disable=broad-exception-caught
                got_value = None
            if got_value is not None:
                return got_value

    return None


def _extract_content_text(content: Any) -> str:
    """Extract text from a qwen-agent Message content field.

    Content can be a plain string or a list of ContentItem.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                texts.append(text)
        return "\n".join(texts)
    return str(content) if content else ""


def _int_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_token_values(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}

    input_tokens = _int_value(
        _field_value(usage, "input_tokens", "prompt_tokens")
    )
    output_tokens = _int_value(
        _field_value(usage, "output_tokens", "completion_tokens")
    )
    cache_read_tokens = _int_value(
        _field_value(usage, "cache_read_input_tokens", "cached_prompt_tokens")
    )
    cache_creation_tokens = _int_value(
        _field_value(usage, "cache_creation_input_tokens")
    )

    for detail_name in ("prompt_tokens_details", "input_tokens_details"):
        details = _field_value(usage, detail_name)
        if details is not None and cache_read_tokens is None:
            cache_read_tokens = _int_value(
                _field_value(details, "cached_tokens")
            )

    values: dict[str, int] = {}
    if input_tokens is not None:
        values["input_tokens"] = input_tokens
    if output_tokens is not None:
        values["output_tokens"] = output_tokens
    if cache_read_tokens is not None and cache_read_tokens > 0:
        values["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None and cache_creation_tokens > 0:
        values["cache_creation_input_tokens"] = cache_creation_tokens

    return values


def _usage_score(usage_values: dict[str, int]) -> int:
    return (usage_values.get("input_tokens") or 0) + (
        usage_values.get("output_tokens") or 0
    )


def _usage_sources(value: Any) -> list[Any]:
    sources: list[Any] = []
    usage = _field_value(value, "usage")
    if usage is not None:
        sources.append(usage)

    extra = _field_value(value, "extra")
    if extra is not None:
        extra_usage = _field_value(extra, "usage", "usage_metadata")
        if extra_usage is not None:
            sources.append(extra_usage)

        service_info = _field_value(extra, "model_service_info")
        if service_info is not None:
            sources.append(service_info)

    service_info = _field_value(value, "model_service_info")
    if service_info is not None:
        sources.append(service_info)

    return sources


def _extract_usage_values(value: Any, depth: int = 0) -> dict[str, int]:
    """Extract token usage from qwen-agent Message/extra/model_service_info."""
    if value is None or depth > 4:
        return {}

    best_values: dict[str, int] = _usage_token_values(value)

    if isinstance(value, (list, tuple)):
        for item in reversed(value):
            item_values = _extract_usage_values(item, depth + 1)
            if _usage_score(item_values) > _usage_score(best_values):
                best_values = item_values
        return best_values

    for source in _usage_sources(value):
        source_values = _extract_usage_values(source, depth + 1)
        if _usage_score(source_values) > _usage_score(best_values):
            best_values = source_values

    return best_values


def apply_usage_to_inference(
    invocation: InferenceInvocation, value: Any
) -> None:
    """Apply qwen-agent token usage metadata to an InferenceInvocation.

    Qwen-Agent stores DashScope responses under
    ``Message.extra["model_service_info"]`` for both streaming and
    non-streaming calls. Streaming chunks carry cumulative usage, so an
    existing value is only replaced when the candidate usage reports at
    least as many tokens as already observed.
    """
    usage_values = _extract_usage_values(value)
    if not usage_values:
        return

    current_score = (invocation.input_tokens or 0) + (
        invocation.output_tokens or 0
    )
    if current_score and _usage_score(usage_values) < current_score:
        return

    if "input_tokens" in usage_values:
        invocation.input_tokens = usage_values["input_tokens"]
    if "output_tokens" in usage_values:
        invocation.output_tokens = usage_values["output_tokens"]
    if "cache_read_input_tokens" in usage_values:
        invocation.cache_read_input_tokens = usage_values[
            "cache_read_input_tokens"
        ]
    if "cache_creation_input_tokens" in usage_values:
        invocation.cache_creation_input_tokens = usage_values[
            "cache_creation_input_tokens"
        ]


def extract_response_id(response: Any) -> str | None:
    """Extract the DashScope request id to use as ``gen_ai.response.id``.

    Qwen-Agent stores the raw DashScope response under
    ``Message.extra["model_service_info"]``, whose ``request_id``
    identifies the completion.
    """
    if not isinstance(response, list):
        response = [response] if response else []
    for msg in reversed(response):
        extra = _field_value(msg, "extra")
        service_info = (
            _field_value(extra, "model_service_info")
            if extra is not None
            else None
        )
        if service_info is None:
            service_info = _field_value(msg, "model_service_info")
        if service_info is not None:
            request_id = _field_value(service_info, "request_id")
            if request_id:
                return str(request_id)
    return None


def find_tool_call_id(messages: Any, tool_name: str) -> str | None:
    """Best-effort lookup of the current tool call's function id.

    ``FnCallAgent`` passes the message history to ``_call_tool``; the call
    being executed corresponds to the first ``function_call`` message for
    ``tool_name`` that has no function-result message answering its
    ``function_id`` yet.
    """
    if not isinstance(messages, list):
        return None

    responded: set[str] = set()
    pending: list[str] = []
    for msg in messages:
        extra = _field_value(msg, "extra")
        function_id = (
            _field_value(extra, "function_id") if extra is not None else None
        )
        if _field_value(msg, "role") in ("function", "tool"):
            if function_id:
                responded.add(str(function_id))
            continue
        function_call = _field_value(msg, "function_call")
        if (
            function_call
            and _field_value(function_call, "name") == tool_name
            and function_id
        ):
            pending.append(str(function_id))

    for function_id in pending:
        if function_id not in responded:
            return function_id
    return None


def _function_call_part(function_call: Any) -> ToolCallRequest:
    name = _field_value(function_call, "name") or ""
    arguments = _field_value(function_call, "arguments") or "{}"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            pass
    return ToolCallRequest(name=name, arguments=arguments, id=None)


def _tool_call_response_id(msg: Any) -> str:
    """Extract a tool call id for a function/tool role message."""
    tool_call_id = _field_value(msg, "id") or ""
    if not tool_call_id:
        extra = _field_value(msg, "extra")
        if extra is not None:
            tool_call_id = _field_value(extra, "function_id") or ""
    if not tool_call_id:
        tool_call_id = _field_value(msg, "name") or ""
    return tool_call_id


def convert_to_input_messages(messages: Any) -> list[InputMessage]:
    """Convert a qwen-agent Message list to GenAI InputMessage objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    input_messages: list[InputMessage] = []
    for msg in messages:
        try:
            role = _field_value(msg, "role") or "user"
            content = _field_value(msg, "content") or ""
            function_call = _field_value(msg, "function_call")

            parts: list[MessagePart] = []

            if function_call:
                parts.append(_function_call_part(function_call))

            # qwen-agent uses role="function" internally, but the DashScope
            # API converts it to role="tool"; handle both.
            if role in ("function", "tool") and content:
                parts.append(
                    ToolCallResponse(
                        id=_tool_call_response_id(msg),
                        response=_extract_content_text(content),
                    )
                )
            elif content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if parts:
                input_messages.append(InputMessage(role=role, parts=parts))
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Error converting input message", exc_info=True)
            continue

    return input_messages


def convert_to_output_messages(messages: Any) -> list[OutputMessage]:
    """Convert qwen-agent response messages to GenAI OutputMessage objects."""
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    output_messages: list[OutputMessage] = []
    for msg in messages:
        try:
            content = _field_value(msg, "content") or ""
            function_call = _field_value(msg, "function_call")

            parts: list[MessagePart] = []
            finish_reason = "stop"

            if function_call:
                parts.append(_function_call_part(function_call))
                finish_reason = "tool_calls"

            if content:
                text = _extract_content_text(content)
                if text:
                    parts.append(Text(content=text))

            if not parts:
                parts.append(Text(content=""))

            output_messages.append(
                OutputMessage(
                    role="assistant",
                    parts=parts,
                    finish_reason=finish_reason,
                )
            )
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug("Error converting output message", exc_info=True)
            continue

    return output_messages


def convert_to_final_output_messages(messages: Any) -> list[OutputMessage]:
    """Convert only the final qwen-agent answer to OutputMessage objects.

    An agent run yields the full response history including intermediate
    tool calls and tool results; the agent's output is the last assistant
    text message.
    """
    if not messages:
        return []

    if not isinstance(messages, list):
        messages = [messages]

    for msg in reversed(messages):
        try:
            role = _field_value(msg, "role") or "assistant"
            function_call = _field_value(msg, "function_call")
            content = _field_value(msg, "content") or ""

            if role in ("function", "tool") or function_call:
                continue

            if _extract_content_text(content):
                return convert_to_output_messages([msg])
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "Error extracting final agent output message", exc_info=True
            )
            continue

    return []


def has_tool_call(messages: Any) -> bool:
    """Whether any message in a qwen-agent response carries a function call."""
    if not isinstance(messages, list):
        messages = [messages] if messages else []
    return any(_field_value(msg, "function_call") for msg in messages)


def get_tool_definitions(
    functions: Any,
) -> list[FunctionToolDefinition] | None:
    """Convert qwen-agent function dicts to FunctionToolDefinition objects."""
    if not functions:
        return None

    tool_definitions: list[FunctionToolDefinition] = []
    for func in functions:
        if not isinstance(func, dict):
            continue
        name = func.get("name")
        if not name:
            continue
        tool_definitions.append(
            FunctionToolDefinition(
                name=name,
                description=func.get("description"),
                parameters=func.get("parameters"),
            )
        )
    return tool_definitions or None


def create_inference_invocation(
    handler: TelemetryHandler,
    llm_instance: Any,
    messages: Any,
    functions: Any,
    extra_generate_cfg: Any,
) -> InferenceInvocation:
    """Create and start an InferenceInvocation for BaseChatModel.chat()."""
    invocation = handler.inference(
        _get_provider_name(llm_instance),
        request_model=getattr(llm_instance, "model", None),
    )

    if isinstance(extra_generate_cfg, dict):
        max_tokens = extra_generate_cfg.get("max_tokens")
        if max_tokens is not None:
            invocation.max_tokens = max_tokens
        temperature = extra_generate_cfg.get("temperature")
        if temperature is not None:
            invocation.temperature = temperature
        top_p = extra_generate_cfg.get("top_p")
        if top_p is not None:
            invocation.top_p = top_p

    invocation.tool_definitions = get_tool_definitions(functions)

    if handler.should_capture_content():
        invocation.input_messages = convert_to_input_messages(messages)

    return invocation


def create_agent_invocation(
    handler: TelemetryHandler,
    agent_instance: Any,
    messages: Any,
) -> AgentInvocation:
    """Create and start an AgentInvocation for Agent.run()."""
    llm_instance = getattr(agent_instance, "llm", None)
    agent_name = (
        getattr(agent_instance, "name", None) or type(agent_instance).__name__
    )

    # invoke_local_agent() intentionally takes no provider: the INTERNAL
    # invoke_agent span describes in-process agent logic, not a provider call.
    invocation = handler.invoke_local_agent(
        request_model=getattr(llm_instance, "model", None)
        if llm_instance is not None
        else None,
        agent_name=agent_name,
    )
    invocation.agent_description = (
        getattr(agent_instance, "description", None) or None
    )

    if handler.should_capture_content():
        invocation.input_messages = convert_to_input_messages(messages)
        # Agent.system_message is the configured system prompt that
        # qwen-agent prepends to the LLM messages on every run.
        system_message = getattr(agent_instance, "system_message", None)
        if system_message:
            invocation.system_instruction = [Text(content=system_message)]

    return invocation
