# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentScope v2 middleware instrumentation."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from typing import Any
from urllib.parse import urlparse

from agentscope.agent import Agent
from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatModelBase, ChatResponse
from agentscope.tool import ToolResponse

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
    Reasoning,
    Text,
    ToolCallRequest,
    ToolCallResponse,
    ToolDefinition,
)

logger = logging.getLogger(__name__)

_MIDDLEWARE_PARAMETER = "middlewares"


def append_agentscope_middleware(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    middleware: "AgentScopeV2Middleware",
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Append the telemetry middleware to AgentScope v2 ``Agent.__init__`` inputs."""
    if _MIDDLEWARE_PARAMETER in kwargs:
        kwargs = dict(kwargs)
        kwargs[_MIDDLEWARE_PARAMETER] = _append_once(
            kwargs.get(_MIDDLEWARE_PARAMETER), middleware
        )
        return args, kwargs

    middleware_position = _middleware_arg_position()
    if middleware_position is not None and len(args) > middleware_position:
        updated_args = list(args)
        updated_args[middleware_position] = _append_once(
            updated_args[middleware_position],
            middleware,
        )
        return tuple(updated_args), kwargs

    kwargs = dict(kwargs)
    kwargs[_MIDDLEWARE_PARAMETER] = [middleware]
    return args, kwargs


def _append_once(
    middlewares: Sequence[MiddlewareBase] | None,
    middleware: "AgentScopeV2Middleware",
) -> list[MiddlewareBase]:
    result = list(middlewares or [])
    if any(isinstance(item, AgentScopeV2Middleware) for item in result):
        return result
    result.append(middleware)
    return result


class AgentScopeV2Middleware(MiddlewareBase):
    """Telemetry adapter for AgentScope v2 middleware hooks."""

    def __init__(self, handler: Callable[[], TelemetryHandler | None]) -> None:
        self._handler = handler

    async def on_reply(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        handler = self._handler()
        if handler is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        invocation = _start_agent_invocation(handler, agent, input_kwargs)
        last_msg = None
        closed = False
        try:
            async for item in next_handler(**input_kwargs):
                if isinstance(item, Msg):
                    last_msg = item
                yield item
        except GeneratorExit:
            invocation.stop()
            closed = True
            raise
        except BaseException as exc:
            invocation.fail(exc)
            closed = True
            raise
        else:
            if last_msg is not None:
                if handler.should_capture_content():
                    invocation.output_messages = [_message_to_output(last_msg)]
                if last_msg.usage is not None:
                    invocation.input_tokens = last_msg.usage.input_tokens
                    invocation.output_tokens = last_msg.usage.output_tokens
            invocation.stop()
            closed = True
        finally:
            if not closed:
                invocation.stop()

    async def on_model_call(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[
            ...,
            Awaitable[ChatResponse | AsyncGenerator[ChatResponse, None]],
        ],
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        model = input_kwargs.get("current_model")
        if not isinstance(model, ChatModelBase):
            return await next_handler(**input_kwargs)

        handler = self._handler()
        if handler is None:
            return await next_handler(**input_kwargs)

        invocation = _start_llm_invocation(handler, model, input_kwargs)
        try:
            result = await next_handler(**input_kwargs)
            if inspect.isasyncgen(result):
                return self._wrap_model_stream(result, invocation, handler)

            _finish_llm_invocation(invocation, result, handler)
            invocation.stop()
            return result
        except BaseException as exc:
            invocation.fail(exc)
            raise

    async def _wrap_model_stream(
        self,
        result: AsyncGenerator[ChatResponse, None],
        invocation: InferenceInvocation,
        handler: TelemetryHandler,
    ) -> AsyncGenerator[ChatResponse, None]:
        last_chunk = None
        closed = False
        try:
            async for chunk in result:
                last_chunk = chunk
                yield chunk
        except GeneratorExit:
            invocation.stop()
            closed = True
            raise
        except BaseException as exc:
            invocation.fail(exc)
            closed = True
            raise
        else:
            _finish_llm_invocation(invocation, last_chunk, handler)
            invocation.stop()
            closed = True
        finally:
            if not closed:
                invocation.stop()

    async def on_acting(
        self,
        agent: Agent,
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        handler = self._handler()
        if handler is None:
            async for item in next_handler(**input_kwargs):
                yield item
            return

        tool_call = input_kwargs.get("tool_call")
        invocation = handler.tool(
            getattr(tool_call, "name", "unknown_tool"),
            tool_call_id=getattr(tool_call, "id", None),
            tool_type="function",
        )
        if invocation.should_capture_content_on_span:
            invocation.arguments = _loads_json(
                getattr(tool_call, "input", None)
            )
        last_item = None
        closed = False
        try:
            async for item in next_handler(**input_kwargs):
                last_item = item
                yield item
        except GeneratorExit:
            invocation.stop()
            closed = True
            raise
        except BaseException as exc:
            invocation.fail(exc)
            closed = True
            raise
        else:
            if invocation.should_capture_content_on_span:
                if isinstance(last_item, ToolResponse):
                    invocation.tool_result = _jsonable(
                        _blocks_to_parts(last_item.content)
                    )
                elif last_item is not None:
                    invocation.tool_result = str(last_item)
            invocation.stop()
            closed = True
        finally:
            if not closed:
                invocation.stop()


def _start_agent_invocation(
    handler: TelemetryHandler,
    agent: Agent,
    input_kwargs: dict[str, Any],
) -> AgentInvocation:
    model = getattr(agent, "model", None)
    request_model = getattr(model, "model", None)
    invocation = handler.invoke_local_agent(
        request_model=request_model,
        agent_name=getattr(agent, "name", "unknown_agent"),
    )
    invocation.provider = _get_provider_name(model)
    session_id = getattr(getattr(agent, "state", None), "session_id", None)
    invocation.agent_id = session_id
    invocation.conversation_id = session_id
    if handler.should_capture_content():
        invocation.input_messages = _messages_to_inputs(
            input_kwargs.get("inputs")
        )
        invocation.system_instruction = [
            Text(content=getattr(agent, "_system_prompt", ""))
        ]
    return invocation


def _start_llm_invocation(
    handler: TelemetryHandler,
    model: ChatModelBase,
    input_kwargs: dict[str, Any],
) -> InferenceInvocation:
    invocation = handler.inference(
        _get_provider_name(model),
        request_model=getattr(model, "model", None),
    )
    server_address = _get_server_address(model)
    if server_address is not None:
        invocation.server_address = server_address
    if handler.should_capture_content():
        invocation.input_messages = _messages_to_inputs(
            input_kwargs.get("messages")
        )
    invocation.tool_definitions = _tool_definitions(input_kwargs.get("tools"))
    parameters = getattr(model, "parameters", None)
    for source in (parameters, input_kwargs):
        _set_if_present(invocation, "temperature", source)
        _set_if_present(invocation, "top_p", source)
        _set_if_present(invocation, "max_tokens", source)
    return invocation


def _finish_llm_invocation(
    invocation: InferenceInvocation,
    response: ChatResponse | None,
    handler: TelemetryHandler,
) -> None:
    if response is None:
        return
    invocation.response_id = getattr(response, "id", None)
    # AgentScope's ChatResponse carries no response model field, so mirror the
    # request model (matching the v1 wrapper).
    invocation.response_model_name = invocation.request_model
    if handler.should_capture_content():
        invocation.output_messages = [_chat_response_to_output(response)]
    else:
        invocation.finish_reasons = [_response_finish_reason(response)]
    usage = getattr(response, "usage", None)
    if usage is not None:
        invocation.input_tokens = getattr(usage, "input_tokens", None)
        invocation.output_tokens = getattr(usage, "output_tokens", None)


def _messages_to_inputs(value: Any) -> list[InputMessage]:
    if value is None:
        return []
    if isinstance(value, Msg):
        return [_message_to_input(value)]
    if isinstance(value, list):
        return [
            _message_to_input(item) for item in value if isinstance(item, Msg)
        ]
    return []


def _message_to_input(msg: Msg) -> InputMessage:
    return InputMessage(role=msg.role, parts=_blocks_to_parts(msg.content))


def _message_to_output(msg: Msg) -> OutputMessage:
    return OutputMessage(
        role=msg.role,
        parts=_blocks_to_parts(msg.content),
        finish_reason="stop",
    )


def _response_finish_reason(response: ChatResponse) -> str:
    if any(
        getattr(block, "type", None) == "tool_call"
        for block in response.content
    ):
        return "tool_calls"
    return "stop"


def _chat_response_to_output(response: ChatResponse) -> OutputMessage:
    return OutputMessage(
        role="assistant",
        parts=_blocks_to_parts(response.content),
        finish_reason=_response_finish_reason(response),
    )


def _blocks_to_parts(blocks: Sequence[Any]) -> list[MessagePart]:
    parts: list[MessagePart] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(Text(content=getattr(block, "text", "")))
        elif block_type == "thinking":
            parts.append(Reasoning(content=getattr(block, "thinking", "")))
        elif block_type == "tool_call":
            parts.append(
                ToolCallRequest(
                    arguments=_loads_json(getattr(block, "input", None)),
                    name=getattr(block, "name", ""),
                    id=getattr(block, "id", None),
                )
            )
        elif block_type == "tool_result":
            parts.append(
                ToolCallResponse(
                    response=_tool_result_response(
                        getattr(block, "output", "")
                    ),
                    id=getattr(block, "id", None),
                )
            )
    return parts


def _tool_result_response(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return _jsonable(_blocks_to_parts(value))
    return value


def _tool_definitions(
    tools: list[dict[str, Any]] | None,
) -> list[ToolDefinition] | None:
    if not tools:
        return None
    definitions: list[ToolDefinition] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        definitions.append(
            FunctionToolDefinition(
                name=function.get("name", ""),
                description=function.get("description"),
                parameters=function.get("parameters"),
            )
        )
    return definitions or None


def _get_provider_name(model: Any) -> str:
    class_name = model.__class__.__name__.lower() if model is not None else ""
    if "dashscope" in class_name:
        return "dashscope"
    if "openai" in class_name:
        return "openai"
    if "anthropic" in class_name:
        return "anthropic"
    if "gemini" in class_name:
        return "gcp.gen_ai"
    if "ollama" in class_name:
        return "ollama"
    return "agentscope"


def _get_server_address(model: Any) -> str | None:
    """Best-effort host for the ``server.address`` attribute on chat spans.

    Prefer an endpoint the model object exposes; DashScope models don't expose
    one, so fall back to the dashscope SDK's global endpoint.
    """
    for attr in ("base_http_api_url", "base_url"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            host = urlparse(value).hostname
            if host:
                return host
    client = getattr(model, "client", None)
    base_url = getattr(client, "base_url", None)
    if base_url:
        host = urlparse(str(base_url)).hostname
        if host:
            return host
    if "dashscope" in model.__class__.__name__.lower():
        try:
            import dashscope  # noqa: PLC0415

            host = urlparse(dashscope.base_http_api_url).hostname
            if host:
                return host
        except Exception:
            return "dashscope.aliyuncs.com"
    return None


def _middleware_arg_position() -> int | None:
    try:
        parameters = list(inspect.signature(Agent.__init__).parameters)
        return parameters.index(_MIDDLEWARE_PARAMETER) - 1
    except (TypeError, ValueError):
        return None


def _loads_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _jsonable(value: Any) -> Any:
    from dataclasses import asdict, is_dataclass  # noqa: PLC0415

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _set_if_present(
    invocation: InferenceInvocation,
    field_name: str,
    source: Any,
) -> None:
    value = (
        source.get(field_name)
        if isinstance(source, dict)
        else getattr(source, field_name, None)
    )
    if value is not None:
        setattr(invocation, field_name, value)
