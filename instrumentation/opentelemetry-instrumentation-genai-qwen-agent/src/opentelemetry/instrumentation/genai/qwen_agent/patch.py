# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for qwen-agent methods.

- ``Agent.run()`` -> ``invoke_agent`` spans. ``Agent.run()`` returns a
  generator that yields the (growing) response message list; the span stays
  open until the caller drains the generator. ``Agent.run_nonstream()`` is
  not wrapped separately — it calls ``run()`` internally, so a single
  ``invoke_agent`` span is produced by this wrapper.
- ``Agent._call_tool()`` -> ``execute_tool`` spans and agent tool counts.
- ``Agent._call_llm()`` -> agent inference counts.

LLM calls made by the agent are intentionally not instrumented here: the
underlying model client library (e.g. the OpenAI SDK) has its own
instrumentation, and wrapping ``BaseChatModel.chat()`` as well would emit
duplicate LLM spans when both are enabled.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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

_AGENT_RUN_CONTEXT_KEY = (
    "opentelemetry.instrumentation.genai.qwen_agent.current_agent_run"
)


@dataclass
class _AgentRunContext:
    agent: Any
    invocation: LocalAgentInvocation | None = None


def _active_invocation(agent: Any) -> LocalAgentInvocation | None:
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
        super().__init__(stream, invocation, record_stream_chunks=False)
        # Agent.run() is a generator, but its invocation is not an LLM stream.
        # Its chunks must not produce gen_ai.client streaming metrics.
        self._self_agent_invocation = invocation
        self._self_capture_content = capture_content
        self._self_last_response: Any = None
        invocation.suspend()

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
    run_context = _AgentRunContext(instance)
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
) -> Any:
    """Count model calls made by the current agent invocation."""
    invocation = _active_invocation(instance)
    if invocation is not None:
        invocation.inference_calls = (invocation.inference_calls or 0) + 1
    return wrapped(*args, **kwargs)


def wrap_agent_call_tool(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``Agent._call_tool()`` producing an ``execute_tool`` span."""
    agent_invocation = _active_invocation(instance)
    if agent_invocation is not None:
        agent_invocation.tool_calls = (agent_invocation.tool_calls or 0) + 1
    bound = bind_arguments(wrapped, args, kwargs)
    tool_name = str(bound.get("tool_name") or "")
    tool_args: Any = bound.get("tool_args")
    tool = getattr(instance, "function_map", {}).get(tool_name)

    invocation = handler.tool(
        tool_name,
        tool_type="function",
    )
    messages_arg: Any = kwargs.get("messages") or bound.get("messages")
    invocation.tool_call_id = find_tool_call_id(messages_arg, tool_name)
    invocation.tool_description = getattr(tool, "description", None)
    if invocation.should_capture_content:
        invocation.arguments = tool_arguments_to_any_value(tool_args)

    try:
        result = wrapped(*args, **kwargs)
    except BaseException as error:
        invocation.fail(error)
        raise

    if invocation.should_capture_content:
        invocation.tool_result = object_to_any_value(result)
    invocation.stop()
    return result


def wrap_fncall_agent_call_tool(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Count unknown tools that FnCallAgent returns without delegating to Agent."""
    tool_name = get_argument("tool_name", wrapped, args, kwargs)
    if tool_name not in instance.function_map:
        invocation = _active_invocation(instance)
        if invocation is not None:
            invocation.tool_calls = (invocation.tool_calls or 0) + 1
    return wrapped(*args, **kwargs)
