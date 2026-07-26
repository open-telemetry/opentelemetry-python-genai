# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for qwen-agent methods.

- ``Agent.run()`` -> ``invoke_agent`` spans. ``Agent.run()`` returns a
  generator that yields the (growing) response message list; the span stays
  open until the caller drains the generator. ``Agent.run_nonstream()`` is
  not wrapped separately — it calls ``run()`` internally, so a single
  ``invoke_agent`` span is produced by this wrapper.
- ``BaseChatModel.chat()`` -> ``chat`` spans, covering both the streaming
  (iterator) and non-streaming (list) return shapes.
- ``Agent._call_tool()`` -> ``execute_tool`` spans.
"""

from __future__ import annotations

from typing import Any, Callable

from opentelemetry.instrumentation.genai.qwen_agent.utils import (
    apply_usage_to_inference,
    convert_to_final_output_messages,
    convert_to_output_messages,
    create_agent_invocation,
    create_inference_invocation,
    extract_response_id,
    find_tool_call_id,
    has_tool_call,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.stream import SyncStreamWrapper


def _finish_inference(
    invocation: InferenceInvocation,
    response: Any,
    capture_content: bool,
) -> None:
    """Populate response attributes on a chat invocation before stop()."""
    if not response:
        return
    apply_usage_to_inference(invocation, response)
    if capture_content:
        invocation.output_messages = convert_to_output_messages(response)
    invocation.response_id = extract_response_id(response)
    invocation.response_model_name = invocation.request_model
    invocation.finish_reasons = (
        ["tool_calls"] if has_tool_call(response) else ["stop"]
    )


class _ChatStreamWrapper(SyncStreamWrapper[Any]):
    """Wraps the iterator returned by a streaming ``BaseChatModel.chat()``.

    Each chunk is the cumulative response message list; the last chunk seen
    is the full response.
    """

    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_last_response: Any = None

    def _process_chunk(self, chunk: Any) -> None:
        apply_usage_to_inference(self._self_invocation, chunk)
        self._self_last_response = chunk

    def _on_stream_end(self) -> None:
        _finish_inference(
            self._self_invocation,
            self._self_last_response,
            self._self_capture_content,
        )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


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
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_last_response: Any = None

    def _process_chunk(self, chunk: Any) -> None:
        self._self_last_response = chunk

    def _on_stream_end(self) -> None:
        if self._self_capture_content and self._self_last_response:
            self._self_invocation.output_messages = (
                convert_to_final_output_messages(self._self_last_response)
            )
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


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


def wrap_chat_model_chat(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    handler: TelemetryHandler,
) -> Any:
    """Wrapper for ``BaseChatModel.chat()`` producing a ``chat`` span."""
    # chat(messages, functions=None, stream=True, delta_stream=False,
    #      extra_generate_cfg=None)
    messages = args[0] if args else kwargs.get("messages", [])
    functions = args[1] if len(args) > 1 else kwargs.get("functions")
    extra_generate_cfg = (
        args[4] if len(args) > 4 else kwargs.get("extra_generate_cfg")
    )

    invocation = create_inference_invocation(
        handler, instance, messages, functions, extra_generate_cfg
    )
    capture_content = handler.should_capture_content()

    try:
        result = wrapped(*args, **kwargs)
    except Exception as error:
        invocation.fail(error)
        raise

    if isinstance(result, list):
        # Non-streaming: result is the full response message list.
        _finish_inference(invocation, result, capture_content)
        invocation.stop()
        return result

    # Streaming: result is an iterator of cumulative response message lists.
    return _ChatStreamWrapper(result, invocation, capture_content)


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
