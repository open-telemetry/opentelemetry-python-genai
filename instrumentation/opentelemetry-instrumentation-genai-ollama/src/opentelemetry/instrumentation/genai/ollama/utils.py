# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract GenAI telemetry from Ollama requests and responses."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    FunctionToolDefinition,
    InputMessage,
    MessagePart,
    OutputMessage,
    Reasoning,
    Role,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

# The GenAI registry has no value for Ollama, so the product name is used.
PROVIDER = "ollama"


def get(obj: Any, name: str) -> Any:
    """Read ``name`` from a mapping or an object, as Ollama accepts both."""
    if isinstance(obj, Mapping):
        return cast(Mapping[str, Any], obj).get(name)
    return getattr(obj, name, None)


def _arg(
    args: tuple[Any, ...], kwargs: Mapping[str, Any], index: int, name: str
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return None


def materialize_chat_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Turn one-shot ``messages`` and ``tools`` iterables into lists.

    Ollama reads them only once, so capturing content from a generator would
    otherwise leave the request empty. The lists are passed on in their place.
    """
    messages = _arg(args, kwargs, 1, "messages")
    if messages is not None and not isinstance(messages, Sequence):
        materialized = list(cast(Iterable[Any], messages))
        if "messages" in kwargs:
            kwargs = {**kwargs, "messages": materialized}
        else:
            args = (*args[:1], materialized, *args[2:])

    tools = kwargs.get("tools")
    if tools is not None and not isinstance(tools, Sequence):
        kwargs = {**kwargs, "tools": list(cast(Iterable[Any], tools))}
    return args, kwargs


def is_empty_embed_request(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> bool:
    """Whether an ``embed`` call has no input, so nothing is embedded."""
    return not _arg(args, kwargs, 1, "input")


def is_model_load_request(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> bool:
    """Whether a ``chat`` call only loads or unloads the model.

    Ollama treats a chat request without messages as a request to load the
    model, or to unload it when ``keep_alive`` is 0; no inference runs.
    """
    return not _arg(args, kwargs, 1, "messages")


def server_address_and_port(instance: Any) -> tuple[str | None, int | None]:
    base_url = getattr(getattr(instance, "_client", None), "base_url", None)
    host = getattr(base_url, "host", None)
    if not isinstance(host, str) or not host:
        return None, None
    # httpx reports no port when the URL uses the scheme's default one.
    port = getattr(base_url, "port", None)
    if not isinstance(port, int) or port == 443:
        return host, None
    return host, port


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        return value
    if isinstance(value, int) and abs(value) <= sys.float_info.max:
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _apply_options(invocation: InferenceInvocation, options: Any) -> None:
    if options is None:
        return
    invocation.temperature = _as_float(get(options, "temperature"))
    invocation.top_p = _as_float(get(options, "top_p"))
    invocation.top_k = _as_int(get(options, "top_k"))
    invocation.seed = _as_int(get(options, "seed"))
    # A negative num_predict means no limit (-1) or fill the context (-2).
    num_predict = _as_int(get(options, "num_predict"))
    if num_predict is not None and num_predict > 0:
        invocation.max_tokens = num_predict
    invocation.presence_penalty = _as_float(get(options, "presence_penalty"))
    invocation.frequency_penalty = _as_float(get(options, "frequency_penalty"))
    stop = get(options, "stop")
    if isinstance(stop, str):
        invocation.stop_sequences = [stop]
    elif isinstance(stop, Iterable):
        invocation.stop_sequences = [
            str(item) for item in cast(Iterable[Any], stop)
        ]


def tool_call_requests(tool_calls: Any) -> list[MessagePart]:
    parts: list[MessagePart] = []
    if not isinstance(tool_calls, Iterable):
        return parts
    for tool_call in cast(Iterable[Any], tool_calls):
        function = get(tool_call, "function")
        parts.append(
            ToolCallRequest(
                id=None,
                name=str(get(function, "name") or ""),
                arguments=get(function, "arguments"),
            )
        )
    return parts


def _input_messages(messages: Iterable[Any]) -> list[InputMessage]:
    result: list[InputMessage] = []
    for message in messages:
        role = get(message, "role")
        if not role:
            continue
        content = get(message, "content")
        parts: list[MessagePart] = []
        if role == Role.TOOL.value:
            parts.append(ToolCallResponse(id=None, response=content))
        else:
            if thinking := get(message, "thinking"):
                parts.append(Reasoning(content=str(thinking)))
            if content:
                parts.append(Text(content=str(content)))
            parts.extend(tool_call_requests(get(message, "tool_calls")))
        if parts:
            result.append(InputMessage(role=str(role), parts=parts))
    return result


def _tool_definitions(tools: Iterable[Any]) -> list[ToolDefinition] | None:
    definitions: list[ToolDefinition] = []
    for tool in tools:
        if callable(tool) and not isinstance(tool, Mapping):
            # Ollama builds the schema of a Python function from its docstring
            # with a private helper, so only the name is recorded.
            definitions.append(
                FunctionToolDefinition(
                    name=getattr(tool, "__name__", ""),
                    description=None,
                    parameters=None,
                )
            )
            continue
        function = get(tool, "function")
        if function is None:
            continue
        parameters = get(function, "parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            dump = getattr(parameters, "model_dump", None)
            parameters = dump(exclude_none=True) if callable(dump) else None
        definitions.append(
            FunctionToolDefinition(
                name=str(get(function, "name") or ""),
                description=get(function, "description"),
                parameters=cast(Mapping[str, Any] | None, parameters),
            )
        )
    return definitions or None


def create_chat_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    capture_content: bool,
) -> InferenceInvocation:
    model = _arg(args, kwargs, 0, "model")
    address, port = server_address_and_port(instance)
    invocation = handler.inference(
        PROVIDER,
        request_model=str(model) if model else None,
        server_address=address,
        server_port=port,
    )
    _apply_options(invocation, kwargs.get("options"))
    if kwargs.get("format"):
        invocation.output_type = (
            GenAIAttributes.GenAiOutputTypeValues.JSON.value
        )

    tools = kwargs.get("tools")
    if isinstance(tools, Iterable):
        invocation.tool_definitions = _tool_definitions(
            cast(Iterable[Any], tools)
        )

    messages = _arg(args, kwargs, 1, "messages")
    if capture_content and isinstance(messages, Iterable):
        invocation.input_messages = _input_messages(
            cast(Iterable[Any], messages)
        )
    return invocation


def finish_reason(done_reason: Any, has_tool_calls: bool) -> str | None:
    """Normalize Ollama's ``done_reason``, which is ``stop`` for tool calls too."""
    if has_tool_calls:
        return "tool_call"
    return str(done_reason) if done_reason else None


def output_message(
    content: str, thinking: str, tool_calls: Sequence[MessagePart]
) -> OutputMessage:
    parts: list[MessagePart] = []
    if thinking:
        parts.append(Reasoning(content=thinking))
    if content:
        parts.append(Text(content=content))
    parts.extend(tool_calls)
    return OutputMessage(role=Role.ASSISTANT.value, parts=parts)


def apply_final_chunk(
    invocation: InferenceInvocation, response: Any, has_tool_calls: bool
) -> None:
    """Record the model, finish reason, and token usage of a chat response."""
    if (model := get(response, "model")) is not None:
        invocation.response_model_name = str(model)
    reason = finish_reason(get(response, "done_reason"), has_tool_calls)
    if reason is not None:
        invocation.finish_reasons = [reason]
    invocation.input_tokens = _as_int(get(response, "prompt_eval_count"))
    invocation.output_tokens = _as_int(get(response, "eval_count"))


def set_chat_response(
    invocation: InferenceInvocation, response: Any, capture_content: bool
) -> None:
    message = get(response, "message")
    tool_calls = tool_call_requests(get(message, "tool_calls"))
    apply_final_chunk(invocation, response, bool(tool_calls))
    if capture_content:
        invocation.output_messages = [
            output_message(
                str(get(message, "content") or ""),
                str(get(message, "thinking") or ""),
                tool_calls,
            )
        ]


def create_embed_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> EmbeddingInvocation:
    model = _arg(args, kwargs, 0, "model")
    address, port = server_address_and_port(instance)
    return handler.embedding(
        PROVIDER,
        request_model=str(model) if model else None,
        server_address=address,
        server_port=port,
    )


def set_embed_response(invocation: EmbeddingInvocation, response: Any) -> None:
    if (model := get(response, "model")) is not None:
        invocation.response_model_name = str(model)
    invocation.input_tokens = _as_int(get(response, "prompt_eval_count"))
    embeddings = get(response, "embeddings")
    if isinstance(embeddings, Sequence) and embeddings:
        first = cast(Sequence[Any], embeddings)[0]
        if isinstance(first, Sequence):
            invocation.dimension_count = len(cast(Sequence[Any], first))
