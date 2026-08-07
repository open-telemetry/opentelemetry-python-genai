# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patching functions for Agno instrumentation."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, cast

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.agno.utils import (
    prepare_tool_definitions,
)
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    TextPart,
)

logger = logging.getLogger(__name__)

_AGNO_MODULE = "agno.agent"
_AGENT_CLASS = "Agent"
_AGNO_TOOLS_MODULE = "agno.tools.function"
_FUNCTION_CALL_CLASS = "FunctionCall"


def patch_agent(handler: TelemetryHandler) -> None:
    """Apply patches to agno.agent.Agent and agno.tools.function.FunctionCall class methods."""
    wrap_function_wrapper(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.run",
        _agent_run(handler),
    )
    wrap_function_wrapper(
        _AGNO_MODULE,
        f"{_AGENT_CLASS}.arun",
        _agent_arun(handler),
    )
    try:
        wrap_function_wrapper(
            _AGNO_TOOLS_MODULE,
            f"{_FUNCTION_CALL_CLASS}.execute",
            _tool_call_execute(handler),
        )
        wrap_function_wrapper(
            _AGNO_TOOLS_MODULE,
            f"{_FUNCTION_CALL_CLASS}.aexecute",
            _tool_call_aexecute(handler),
        )
    except (ImportError, AttributeError):
        pass


def unpatch_agent() -> None:
    """Remove patches from agno.agent.Agent class methods."""
    try:
        import agno.agent  # pylint: disable=import-outside-toplevel

        unwrap(agno.agent.Agent, "run")
        unwrap(agno.agent.Agent, "arun")
    except (ImportError, AttributeError):
        pass
    try:
        import agno.tools.function  # pylint: disable=import-outside-toplevel

        unwrap(agno.tools.function.FunctionCall, "execute")
        unwrap(agno.tools.function.FunctionCall, "aexecute")
    except (ImportError, AttributeError):
        pass


def _extract_input_content(input_val: Any) -> str:
    if isinstance(input_val, str):
        return input_val
    if hasattr(input_val, "content"):
        return str(getattr(input_val, "content"))
    return str(input_val)


def _extract_output_content(result: Any) -> str:
    if result is None:
        return ""
    if hasattr(result, "content") and getattr(result, "content") is not None:
        content = getattr(result, "content")
        if isinstance(content, str):
            return content
        return str(content)
    if hasattr(result, "result") and getattr(result, "result") is not None:
        val = getattr(result, "result")
        if isinstance(val, str):
            return val
        return str(val)
    return str(result)


def _extract_arguments_str(args_val: Any) -> str:
    if isinstance(args_val, str):
        return args_val
    if isinstance(args_val, (dict, list)):
        try:
            return json.dumps(args_val, ensure_ascii=False)
        except Exception:
            pass
    return str(cast(object, args_val))


def _set_tool_invocation_input(
    invocation: ToolInvocation,
    instance: Any,
    capture_content: bool,
) -> None:
    if capture_content:
        arguments = getattr(instance, "arguments", None)
        if arguments is not None:
            invocation.arguments = _extract_arguments_str(arguments)


def _set_tool_invocation_output(
    invocation: Any,
    result: Any,
    capture_content: bool,
) -> None:
    if capture_content and result is not None:
        invocation.tool_result = _extract_output_content(result)


def _set_invocation_input(
    invocation: AgentInvocation,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> None:
    agent_id = getattr(instance, "agent_id", None)
    if agent_id:
        invocation.agent_id = str(agent_id)

    if capture_content and (args or "input" in kwargs):
        input_val = args[0] if args else kwargs.get("input")
        if input_val is not None:
            content_str = _extract_input_content(input_val)
            invocation.input_messages = [
                InputMessage(
                    role="user", parts=[TextPart(content=content_str)]
                )
            ]


def _set_invocation_output(
    invocation: Any,
    result: Any,
    capture_content: bool,
) -> None:
    if capture_content and result is not None:
        output_str = _extract_output_content(result)
        invocation.output_messages = [
            OutputMessage(
                role="assistant",
                parts=[TextPart(content=output_str)],
                finish_reason="stop",
            )
        ]
    if hasattr(result, "session_id") and getattr(result, "session_id"):
        invocation.conversation_id = str(getattr(result, "session_id"))


def _start_agent_invocation(
    handler: TelemetryHandler,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    capture_content: bool,
) -> AgentInvocation:
    agent_name = getattr(instance, "name", None) or "Agent"
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    _set_invocation_input(invocation, instance, args, kwargs, capture_content)
    invocation.tool_definitions = prepare_tool_definitions(
        getattr(instance, "tools", None)
    )
    return invocation


def _start_tool_invocation(
    handler: TelemetryHandler,
    instance: Any,
    capture_content: bool,
) -> ToolInvocation:
    function_obj = getattr(instance, "function", None)
    tool_name = getattr(function_obj, "name", None) or "tool"
    tool_desc = getattr(function_obj, "description", None)
    tool_call_id = getattr(instance, "call_id", None)

    invocation = handler.tool(
        name=str(tool_name),
        tool_call_id=str(tool_call_id) if tool_call_id else None,
        tool_type="function",
        tool_description=str(tool_desc) if tool_desc else None,
    )
    _set_tool_invocation_input(invocation, instance, capture_content)
    return invocation


def _agent_run(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with _start_agent_invocation(
            handler, instance, args, kwargs, capture_content
        ) as invocation:
            result = wrapped(*args, **kwargs)
            _set_invocation_output(invocation, result, capture_content)
            return result

    return traced_method


def _agent_arun(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with _start_agent_invocation(
            handler, instance, args, kwargs, capture_content
        ) as invocation:
            result = await wrapped(*args, **kwargs)
            _set_invocation_output(invocation, result, capture_content)
            return result

    return cast(Callable[..., Any], traced_method)


def _tool_call_execute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    def traced_method(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with _start_tool_invocation(
            handler, instance, capture_content
        ) as invocation:
            result = wrapped(*args, **kwargs)
            _set_tool_invocation_output(invocation, result, capture_content)
            return result

    return traced_method


def _tool_call_aexecute(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    capture_content = handler.should_capture_content()

    async def traced_method(
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with _start_tool_invocation(
            handler, instance, capture_content
        ) as invocation:
            result = await wrapped(*args, **kwargs)
            _set_tool_invocation_output(invocation, result, capture_content)
            return result

    return cast(Callable[..., Any], traced_method)
