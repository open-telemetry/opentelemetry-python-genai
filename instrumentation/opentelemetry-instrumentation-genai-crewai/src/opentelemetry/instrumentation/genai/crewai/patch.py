# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Monkey-patch wrappers for CrewAI agent and tool execution.

Each public factory takes the shared ``TelemetryHandler`` and returns a
``wrapt``-style wrapper ``(wrapped, instance, args, kwargs)`` that opens the
matching GenAI invocation around the original CrewAI method. The wrappers
observe only: the original call is made exactly once, its return value and
exceptions pass through unchanged, and telemetry extraction failures are
logged at debug level instead of surfacing to the application.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
from collections.abc import Callable, Iterator
from typing import Any, cast

from crewai.agent.core import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.telemetry.telemetry import Telemetry
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.types import LLMMessage

from opentelemetry.context import (
    _SUPPRESS_INSTRUMENTATION_KEY,
    get_value,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation

from ._messages import (
    CrewAITool,
    agent_tool_definitions,
    bind_call_arguments,
    crewai_tools,
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
"""Name of the agent whose invocation is currently open, or ``None``.

Set by the agent wrappers and read by the tool wrapper to stamp
``gen_ai.agent.name`` on ``execute_tool`` spans, since CrewAI's tool
entry points carry no reference to the calling agent. It propagates into
CrewAI's thread pools because CrewAI copies the context into them.
"""


def _suppressed() -> bool:
    """Return whether instrumentation is suppressed in the current context."""
    return bool(get_value(_SUPPRESS_INSTRUMENTATION_KEY))


def _set_metadata_safely(callback: Callable[[], None]) -> None:
    """Run a telemetry-extraction callback, swallowing any exception.

    Only telemetry extraction belongs in ``callback``; the wrapped
    application call must stay outside so its errors propagate.

    Args:
        callback: Zero-argument callable that sets invocation attributes.
    """
    try:
        callback()
    except Exception:
        _logger.debug("Failed to collect CrewAI metadata", exc_info=True)


def _request_model(agent: Agent) -> str | None:
    """Return the model name of an agent's LLM.

    Args:
        agent: CrewAI ``Agent`` whose ``llm.model`` is read.

    Returns:
        The model name, or ``None`` if the agent's LLM has not been resolved
        to a ``BaseLLM`` instance or has no model.
    """
    llm = agent.llm
    if isinstance(llm, BaseLLM):
        return llm.model or None
    return None


@contextlib.contextmanager
def _agent_invocation(
    handler: TelemetryHandler,
    agent: Agent,
    collect_input: Callable[[LocalAgentInvocation], None],
) -> Iterator[LocalAgentInvocation]:
    """Open an ``invoke_agent`` invocation for a CrewAI agent.

    The invocation is named after the agent's ``role`` and carries its
    ``goal`` as description and its LLM model as request model. While the
    body runs, ``_current_agent_name`` names this agent so tool spans can be
    attributed to it. The invocation is finalized by the handler's context
    manager on exit, as success or with the raised exception.

    Args:
        handler: Telemetry handler used to create the invocation.
        agent: The agent being invoked.
        collect_input: Sets call-specific attributes (tool definitions,
            input messages) on the invocation; failures are logged, not
            raised.

    Yields:
        The open invocation, for recording the output.
    """
    agent_name = agent.role
    with handler.invoke_local_agent(
        agent_name=agent_name, request_model=_request_model(agent)
    ) as invocation:

        def collect() -> None:
            invocation.agent_description = agent.goal or None
            collect_input(invocation)

        _set_metadata_safely(collect)
        token = _current_agent_name.set(agent_name)
        try:
            yield invocation
        finally:
            _current_agent_name.reset(token)


def _record_agent_output(
    invocation: LocalAgentInvocation, result: object
) -> None:
    """Record an agent's return value as the invocation output.

    Args:
        invocation: The open agent invocation.
        result: The value returned by the wrapped CrewAI method.
    """
    if invocation.should_capture_content:
        _set_metadata_safely(
            lambda: setattr(
                invocation,
                "output_messages",
                output_to_output_messages(result),
            )
        )


def agent_execute_task(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build the wrapper for ``Agent.execute_task(task, context, tools)``.

    The wrapper opens an ``invoke_agent`` invocation named after the agent's
    ``role``, records the agent's ``goal`` as the description, the ``llm``
    model as the request model, and the tools passed to the call (falling back
    to ``agent.tools`` when omitted) as tool definitions. With content capture
    enabled, the task description, expected output and context become the
    input message and the returned value the output message.

    Every method invocation gets its own span: a retry that re-enters
    ``execute_task`` from ``_handle_execution_error`` produces a nested span,
    and a retry driven by a task guardrail produces a sibling.

    Args:
        handler: Telemetry handler used to create the invocation.

    Returns:
        A ``wrapt`` wrapper ``(wrapped, instance, args, kwargs) -> Any`` that
        returns whatever the original method returns.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Agent,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)

        def collect_input(invocation: LocalAgentInvocation) -> None:
            bound = bind_call_arguments(wrapped, args, kwargs)
            tools: list[CrewAITool] | None = crewai_tools(bound.get("tools"))
            if tools is None:
                tools = crewai_tools(instance.tools)
            invocation.tool_definitions = agent_tool_definitions(tools)
            task: object = bound.get("task")
            context: object = bound.get("context")
            if invocation.should_capture_content and isinstance(task, Task):
                invocation.input_messages = task_to_input_messages(
                    task,
                    context=context if isinstance(context, str) else None,
                )

        with _agent_invocation(handler, instance, collect_input) as invocation:
            result = wrapped(*args, **kwargs)
            _record_agent_output(invocation, result)
            return result

    return wrapper


def agent_kickoff(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build the wrapper for standalone ``Agent.kickoff(messages, ...)``.

    Standalone execution does not go through ``execute_task``, so it gets its
    own ``invoke_agent`` invocation with the same agent attributes. With
    content capture enabled, the ``messages`` argument (a string or a list
    of role/content messages) becomes the input and ``LiteAgentOutput.raw``
    the output.

    When an event loop is already running (for example inside a ``Flow``),
    CrewAI returns a coroutine from ``kickoff`` instead of executing. The
    coroutine is wrapped so the invocation opens when it is awaited and
    closes when it completes, rather than around the un-started coroutine.

    Args:
        handler: Telemetry handler used to create the invocation.

    Returns:
        A ``wrapt`` wrapper ``(wrapped, instance, args, kwargs) -> Any`` that
        returns whatever the original method returns.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: Agent,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)

        def collect_input(invocation: LocalAgentInvocation) -> None:
            invocation.tool_definitions = agent_tool_definitions(
                crewai_tools(instance.tools)
            )
            if invocation.should_capture_content:
                bound = bind_call_arguments(wrapped, args, kwargs)
                messages: object = bound.get("messages")
                if isinstance(messages, (str, list)):
                    invocation.input_messages = messages_to_input_messages(
                        cast("str | list[LLMMessage]", messages)
                    )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            with _agent_invocation(
                handler, instance, collect_input
            ) as invocation:
                result = wrapped(*args, **kwargs)
                _record_agent_output(invocation, result)
                return result

        # Calling kickoff here only builds the kickoff_async coroutine; the
        # agent runs when the caller awaits it.
        awaitable = wrapped(*args, **kwargs)
        if not inspect.isawaitable(awaitable):
            return awaitable

        async def traced() -> object:
            with _agent_invocation(
                handler, instance, collect_input
            ) as invocation:
                result: object = await awaitable
                _record_agent_output(invocation, result)
                return result

        return traced()

    return wrapper


def crewai_telemetry_disabled(
    wrapped: Callable[..., Any],
    instance: Telemetry,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """Wrapper for ``Telemetry._safe_telemetry_operation`` that drops the call.

    Every CrewAI usage-telemetry emitter funnels through that method, and
    returning ``None`` is the branch it takes under
    ``CREWAI_DISABLE_TELEMETRY``, so callers already handle it.

    Args:
        wrapped: The original method; never called.
        instance: The ``Telemetry`` singleton; unused.
        args: Positional arguments of the original call; unused.
        kwargs: Keyword arguments of the original call; unused.

    Returns:
        Always ``None``, which CrewAI treats as telemetry being disabled.
    """
    del wrapped, instance, args, kwargs


def tool_execution(handler: TelemetryHandler) -> Callable[..., Any]:
    """Build the ``execute_tool`` wrapper for both CrewAI tool entry points.

    ``BaseTool.run`` covers direct calls and CrewAI's native function-calling
    path; ``CrewStructuredTool.invoke`` covers the ReAct path, where
    ``ToolUsage`` calls the adapter directly and ``run`` is bypassed. The
    invocation is named after the tool's ``name``, typed ``function``, and
    stamped with the agent name from ``_current_agent_name`` when a tool runs
    inside an agent invocation. With content capture enabled, the call
    arguments (bound against ``BaseTool._run``, or taken from the adapter's
    ``input`` payload without its ``config``) and the return value are
    recorded as well.

    Args:
        handler: Telemetry handler used to create the invocation.

    Returns:
        A ``wrapt`` wrapper ``(wrapped, instance, args, kwargs) -> Any`` that
        returns whatever the original method returns.
    """

    def wrapper(
        wrapped: Callable[..., Any],
        instance: CrewAITool,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        if _suppressed():
            return wrapped(*args, **kwargs)

        with handler.tool(
            name=instance.name,
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
                        if isinstance(instance, CrewStructuredTool)
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
