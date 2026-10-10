# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``AgentRunner.query_handler`` with an ``invoke_agent`` invocation.

``query_handler`` is an async generator: the invocation must stay open until
the caller drains (or closes) the stream, so the returned generator is
proxied through :class:`QueryHandlerStreamWrapper` which finalizes the
telemetry exactly once on success, error, or close.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from typing import cast

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation
from opentelemetry.util.genai.stream import AsyncStreamWrapper
from opentelemetry.util.genai.types import OutputMessage

from .utils import (
    input_messages_from_msgs,
    non_empty_str,
    output_message_from_yield_item,
    parse_query_handler_call,
)


class QueryHandlerStreamWrapper(AsyncStreamWrapper[object]):
    """Proxy the ``query_handler`` async generator and finalize telemetry."""

    def __init__(
        self,
        stream: AsyncGenerator[object, None],
        invocation: LocalAgentInvocation,
    ) -> None:
        super().__init__(stream)
        # INTERNAL agent invocations must not opt into the base wrapper's
        # streamed inference timing metrics or gen_ai.request.stream handling.
        self._self_agent_invocation = invocation
        self._self_output_complete = False
        self._self_output_message: OutputMessage | None = None

    def _process_chunk(self, chunk: object) -> None:
        self._self_output_complete = False
        if not isinstance(chunk, tuple):
            return
        item = cast("tuple[object, ...]", chunk)
        self._self_output_complete = (
            len(item) > 1
            and getattr(item[0], "role", None) == "assistant"
            and item[1] is True
        )
        if not self._self_agent_invocation.should_capture_content:
            return
        output_message = output_message_from_yield_item(item)
        if output_message is not None:
            self._self_output_message = output_message

    def _on_stream_end(self) -> None:
        self._apply_output_message()
        if self._self_output_complete:
            self._self_agent_invocation.finish_reasons = ["stop"]
        self._self_agent_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._apply_output_message()
        self._self_agent_invocation.fail(error)

    def _apply_output_message(self) -> None:
        if self._self_output_message is not None:
            self._self_agent_invocation.output_messages = [
                self._self_output_message
            ]


def _build_invocation(
    handler: TelemetryHandler,
    instance: object,
    msgs: object,
    request: object,
) -> LocalAgentInvocation:
    # The public agent_name property lazily loads configuration from disk.
    agent_name = non_empty_str(getattr(instance, "_agent_name", None))
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    # The runner's `agent_id` is a local config key (e.g. "default"), not a
    # provider-assigned stable identifier, so `gen_ai.agent.id` is not
    # recorded per its semconv guidance.
    conversation_id = non_empty_str(getattr(request, "session_id", None))
    if conversation_id is not None:
        invocation.conversation_id = conversation_id
    if invocation.should_capture_content:
        invocation.input_messages = input_messages_from_msgs(msgs)
    return invocation


def make_query_handler_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., QueryHandlerStreamWrapper]:
    """Factory for the ``wrapt`` wrapper bound to *handler*."""

    def query_handler_wrapper(
        wrapped: Callable[..., AsyncGenerator[object, None]],
        # QwenPaw cannot be imported for typing on Python >= 3.14.
        instance: object,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> QueryHandlerStreamWrapper:
        msgs, request = parse_query_handler_call(args, kwargs)
        invocation = _build_invocation(handler, instance, msgs, request)
        try:
            stream = wrapped(*args, **kwargs)
        except BaseException as exc:
            invocation.fail(exc)
            raise
        return QueryHandlerStreamWrapper(stream, invocation)

    return query_handler_wrapper
