# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for qwen-agent methods.

- ``Agent.run()`` -> ``invoke_agent`` spans. ``Agent.run()`` returns a
  generator that yields the (growing) response message list; the span stays
  open until the caller drains the generator. ``Agent.run_nonstream()`` is
  not wrapped separately — it calls ``run()`` internally, so a single
  ``invoke_agent`` span is produced by this wrapper.
- ``Agent._call_tool()`` -> ``execute_tool`` spans.
- ``Agent._call_llm()`` and ``Agent._call_tool()`` also count the logical
  model calls and tool calls made by the run, reported as the
  ``gen_ai.invoke_agent.inference_calls`` and
  ``gen_ai.invoke_agent.tool_calls`` histograms when the invocation ends.
  ``FnCallAgent._call_tool()`` answers unknown tools without delegating to
  ``Agent._call_tool()``, so it is wrapped as well to keep those calls
  counted exactly once.

LLM calls made by the agent are intentionally not instrumented here: the
underlying model client library (e.g. the OpenAI SDK) has its own
instrumentation, and wrapping ``BaseChatModel.chat()`` as well would emit
duplicate LLM spans when both are enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry.context import get_value, set_value
from opentelemetry.instrumentation.genai.qwen_agent.utils import (
    convert_to_final_output_messages,
    create_agent_invocation,
    find_tool_call_id,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation
from opentelemetry.util.genai.stream import SyncStreamWrapper
from opentelemetry.util.genai.utils import (
    bind_arguments,
    get_argument,
    object_to_any_value,
    tool_arguments_to_any_value,
)

if TYPE_CHECKING:
    from qwen_agent.agent import Agent

_AGENT_RUN_CONTEXT_KEY = (
    "opentelemetry.instrumentation.genai.qwen_agent.current_agent_run"
)


@dataclass
class _AgentRunContext:
    """Binds an agent's current run to the invocation counting its calls.

    The context is carried by the invocation's own OpenTelemetry context, so
    a nested sub-agent run shadows the parent's entry and both agents keep
    separate counts.
    """

    agent: Agent
    invocation: LocalAgentInvocation | None = None


def _active_invocation(agent: Agent) -> LocalAgentInvocation | None:
    """Return the invocation of the run currently executing for ``agent``."""
    run_context = get_value(_AGENT_RUN_CONTEXT_KEY)
    if (
        isinstance(run_context, _AgentRunContext)
        and run_context.agent is agent
    ):
        return run_context.invocation
    return None


class _AgentRunStreamWrapper(SyncStreamWrapper[Any]):
    """Wraps the generator returned by ``Agent.run()``.

    Each yielded item is the (growing) response message list; the last item
    contains the full run output.
    """

    def __init__(
        self,
        stream: Any,
        invocation: LocalAgentInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream, invocation)
        self._self_agent_invocation = invocation
        self._self_capture_content = capture_content
        self._self_last_response: Any = None

    def _process_chunk(self, chunk: Any) -> None:
        self._self_last_response = chunk

    def _on_stream_end(self) -> None:
        if self._self_capture_content and self._self_last_response:
            self._self_agent_invocation.output_messages = (
                convert_to_final_output_messages(self._self_last_response)
            )
        self._self_agent_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_agent_invocation.fail(error)


def wrap_agent_run(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``Agent.run()`` producing an ``invoke_agent`` span."""
    # Agent.run() is a generator function; calling it never raises.
    result = wrapped(*args, **kwargs)
    messages: Any = get_argument("messages", wrapped, args, kwargs, default=[])
    run_context = _AgentRunContext(agent=instance)
    invocation = create_agent_invocation(
        handler,
        instance,
        messages,
        context=set_value(_AGENT_RUN_CONTEXT_KEY, run_context),
    )
    run_context.invocation = invocation
    invocation.inference_calls = 0
    invocation.tool_calls = 0
    return _AgentRunStreamWrapper(
        result, invocation, handler.should_capture_content()
    )


def wrap_agent_call_llm(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``Agent._call_llm()`` counting model calls of the run.

    The wrapper counts the agent's logical model calls, not the retries the
    model client performs inside ``chat()``.
    """
    invocation = _active_invocation(instance)
    if invocation is not None and invocation.inference_calls is not None:
        invocation.inference_calls += 1
    return wrapped(*args, **kwargs)


def wrap_agent_call_tool(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``Agent._call_tool()`` producing an ``execute_tool`` span."""
    invocation = _active_invocation(instance)
    if invocation is not None and invocation.tool_calls is not None:
        invocation.tool_calls += 1

    bound = bind_arguments(wrapped, args, kwargs)
    tool_name = str(bound.get("tool_name") or "")
    tool_args: Any = bound.get("tool_args")
    tool = getattr(instance, "function_map", {}).get(tool_name)

    invocation = handler.tool(
        tool_name,
        tool_type="function",
        _attach_to_context=False,
    )
    messages_arg: Any = kwargs.get("messages") or bound.get("messages")
    invocation.tool_call_id = find_tool_call_id(messages_arg, tool_name)
    invocation.tool_description = getattr(tool, "description", None)
    if invocation.should_capture_content:
        invocation.arguments = tool_arguments_to_any_value(tool_args)

    with invocation:
        result = wrapped(*args, **kwargs)
        if invocation.should_capture_content:
            invocation.tool_result = object_to_any_value(result)
    return result


def wrap_fncall_agent_call_tool(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``FnCallAgent._call_tool()`` counting unknown tools.

    Known tools are delegated to ``Agent._call_tool()`` and counted there;
    an unknown tool returns early, so it is counted here to avoid missing it.
    """
    tool_name = str(get_argument("tool_name", wrapped, args, kwargs) or "")
    if tool_name not in instance.function_map:
        invocation = _active_invocation(instance)
        if invocation is not None and invocation.tool_calls is not None:
            invocation.tool_calls += 1
    return wrapped(*args, **kwargs)
