# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for qwen-agent methods.

- ``Agent.run()`` -> ``invoke_agent`` spans. ``Agent.run()`` returns a
  generator that yields the (growing) response message list; the span stays
  open until the caller drains the generator. ``Agent.run_nonstream()`` is
  not wrapped separately — it calls ``run()`` internally, so a single
  ``invoke_agent`` span is produced by this wrapper.
- ``Agent._call_tool()`` -> ``execute_tool`` spans.

LLM calls made by the agent are intentionally not instrumented here: the
underlying model client library (e.g. the OpenAI SDK) has its own
instrumentation, and wrapping ``BaseChatModel.chat()`` as well would emit
duplicate LLM spans when both are enabled.
"""

from __future__ import annotations

from typing import Any, Callable

from opentelemetry.instrumentation.genai.qwen_agent.utils import (
    convert_to_final_output_messages,
    create_agent_invocation,
    find_tool_call_id,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation
from opentelemetry.util.genai.stream import SyncStreamWrapper


class _AgentRunStreamWrapper(SyncStreamWrapper[Any]):
    """Wraps the generator returned by ``Agent.run()``.

    Each yielded item is the (growing) response message list; the last item
    contains the full run output.
    """

    def __init__(
        self,
        stream: Any,
        invocation: AgentInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        # Deliberately not passed to super().__init__: the base class would
        # mark the invocation as streamed and record per-chunk LLM streaming
        # metrics, which don't apply to a local invoke_agent invocation.
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
    messages = args[0] if args else kwargs.get("messages", [])
    invocation = create_agent_invocation(handler, instance, messages)
    return _AgentRunStreamWrapper(
        result, invocation, handler.should_capture_content()
    )


def wrap_agent_call_tool(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``Agent._call_tool()`` producing an ``execute_tool`` span."""
    tool_name = str(args[0]) if args else str(kwargs.get("tool_name", ""))
    tool_args = args[1] if len(args) > 1 else kwargs.get("tool_args")
    tool = getattr(instance, "function_map", {}).get(tool_name)

    invocation = handler.tool(
        tool_name,
        tool_call_id=find_tool_call_id(kwargs.get("messages"), tool_name),
        tool_type="function",
        tool_description=getattr(tool, "description", None),
    )
    if invocation.should_capture_content_on_span and tool_args is not None:
        invocation.arguments = tool_args

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        invocation.fail(error)
        raise

    if invocation.should_capture_content_on_span and result is not None:
        invocation.tool_result = (
            result if isinstance(result, str) else str(result)
        )
    invocation.stop()
    return result
