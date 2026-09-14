# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch wrappers for CrewAI agent and tool execution."""

from __future__ import annotations

import asyncio
import contextvars
import logging
from collections.abc import Callable
from typing import Any

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_value,
)
from opentelemetry.util.genai.handler import TelemetryHandler

from ._messages import (
    agent_tool_definitions,
    bind_call_arguments,
    messages_to_input_messages,
    output_to_output_messages,
    structured_tool_input_to_arguments,
    task_to_input_messages,
    tool_args_to_arguments,
    tool_description,
    tool_result_to_result,
)

_logger = logging.getLogger(__name__)

_current_agent_name: contextvars.ContextVar[str | None] = (
    contextvars.ContextVar("otel_genai_crewai_agent_name", default=None)
)


def _suppressed() -> bool:
    return bool(get_value(_SUPPRESS_INSTRUMENTATION_KEY))


def _set_metadata_safely(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception:
        _logger.debug("Failed to collect CrewAI metadata", exc_info=True)


def _safe_name(value: Any, attribute: str) -> str:
    try:
        return str(getattr(value, attribute, None) or type(value).__name__)
    except Exception:
        return type(value).__name__


def _request_model(agent: Any) -> str | None:
    try:
        model = getattr(getattr(agent, "llm", None), "model", None)
        return str(model) if model else None
    except Exception:
        return None


def agent_execute_task(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``Agent.execute_task(task, context=None, tools=None)``."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)

        agent_name = _safe_name(instance, "role")
        with handler.invoke_local_agent(
            agent_name=agent_name,
            request_model=_request_model(instance),
        ) as invocation:

            def collect_input() -> None:
                bound = bind_call_arguments(wrapped, args, kwargs)
                tools = bound.get("tools")
                if tools is None:
                    tools = getattr(instance, "tools", None)
                goal = getattr(instance, "goal", None)
                invocation.agent_description = str(goal) if goal else None
                invocation.tool_definitions = agent_tool_definitions(tools)
                if invocation.should_capture_content:
                    invocation.input_messages = task_to_input_messages(
                        bound.get("task"), context=bound.get("context")
                    )

            _set_metadata_safely(collect_input)
            token = _current_agent_name.set(agent_name)
            try:
                result = wrapped(*args, **kwargs)
                if invocation.should_capture_content:
                    _set_metadata_safely(
                        lambda: setattr(
                            invocation,
                            "output_messages",
                            output_to_output_messages(result),
                        )
                    )
                return result
            finally:
                _current_agent_name.reset(token)

    return wrapper


def agent_kickoff(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap synchronous standalone ``Agent.kickoff(messages, ...)`` calls."""

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return wrapped(*args, **kwargs)

        agent_name = _safe_name(instance, "role")
        with handler.invoke_local_agent(
            agent_name=agent_name,
            request_model=_request_model(instance),
        ) as invocation:

            def collect_input() -> None:
                goal = getattr(instance, "goal", None)
                invocation.agent_description = str(goal) if goal else None
                invocation.tool_definitions = agent_tool_definitions(
                    getattr(instance, "tools", None)
                )
                if invocation.should_capture_content:
                    bound = bind_call_arguments(wrapped, args, kwargs)
                    invocation.input_messages = messages_to_input_messages(
                        bound.get("messages")
                    )

            _set_metadata_safely(collect_input)
            token = _current_agent_name.set(agent_name)
            try:
                result = wrapped(*args, **kwargs)
                if invocation.should_capture_content:
                    _set_metadata_safely(
                        lambda: setattr(
                            invocation,
                            "output_messages",
                            output_to_output_messages(result),
                        )
                    )
                return result
            finally:
                _current_agent_name.reset(token)

    return wrapper


def tool_run(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``BaseTool.run(*args, **kwargs)``."""
    return _tool_wrapper(handler, structured=False)


def structured_tool_invoke(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``CrewStructuredTool.invoke(input, config=None, **kwargs)``."""
    return _tool_wrapper(handler, structured=True)


def _tool_wrapper(
    handler: TelemetryHandler, *, structured: bool
) -> Callable[..., Any]:
    def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)

        with handler.tool(
            name=_safe_name(instance, "name"),
            tool_type="function",
            agent_name=_current_agent_name.get(),
        ) as invocation:

            def collect_input() -> None:
                invocation.tool_description = tool_description(instance)
                if invocation.should_capture_content:
                    invocation.arguments = (
                        structured_tool_input_to_arguments(
                            wrapped, args, kwargs
                        )
                        if structured
                        else tool_args_to_arguments(instance, args, kwargs)
                    )

            _set_metadata_safely(collect_input)
            result = wrapped(*args, **kwargs)
            if invocation.should_capture_content:
                _set_metadata_safely(
                    lambda: setattr(
                        invocation,
                        "tool_result",
                        tool_result_to_result(result),
                    )
                )
            return result

    return wrapper
