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
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    LocalAgentInvocation,
)
from opentelemetry.util.genai.stream import AsyncAgentStreamWrapper

from .utils import (
    input_messages_from_msgs,
    non_empty_str,
    output_message_from_yield_item,
    parse_query_handler_call,
)


class QueryHandlerStreamWrapper(AsyncAgentStreamWrapper[object]):
    """Proxy the ``query_handler`` async generator and finalize telemetry."""

    def __init__(
        self,
        stream: AsyncGenerator[object, None],
        invocation: LocalAgentInvocation,
    ) -> None:
        super().__init__(stream, invocation)
        self._self_output_complete = False

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
        if not self._self_invocation.should_capture_content:
            return
        output_message = output_message_from_yield_item(item)
        if output_message is not None:
            self._set_output_messages([output_message])

    def _on_stream_end(self) -> None:
        invocation = self._self_invocation
        if self._self_output_complete and isinstance(
            invocation, AgentInvocation
        ):
            invocation.finish_reasons = ["stop"]
        super()._on_stream_end()


def _build_invocation(
    handler: TelemetryHandler,
    instance: object,
    msgs: object,
    request: object,
) -> LocalAgentInvocation:
    # The public agent_name property lazily loads configuration from disk.
    agent_name = non_empty_str(getattr(instance, "_agent_name", None))
    # The runner's agent_id is a local config key, not a provider-assigned ID.
    invocation = handler.invoke_local_agent(
        agent_name=agent_name,
        conversation_id=non_empty_str(getattr(request, "session_id", None)),
    )
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
