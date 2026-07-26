# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``AgentRunner.query_handler`` with an ``invoke_agent`` invocation.

``query_handler`` is an async generator: the invocation must stay open until
the caller drains (or closes) the stream, so the returned generator is
proxied through :class:`QueryHandlerStreamWrapper` which finalizes the
telemetry exactly once on success, error, or close.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable, cast

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import AgentInvocation
from opentelemetry.util.genai.stream import AsyncStreamWrapper
from opentelemetry.util.genai.types import OutputMessage

from .utils import (
    input_messages_from_msgs,
    non_empty_str,
    output_message_from_yield_item,
    parse_query_handler_call,
)


class QueryHandlerStreamWrapper(AsyncStreamWrapper[Any]):
    """Proxy the ``query_handler`` async generator and finalize telemetry."""

    def __init__(
        self,
        stream: AsyncGenerator[Any, None],
        invocation: AgentInvocation,
        capture_content: bool,
    ) -> None:
        # Async generators expose ``aclose()`` rather than the ``close()``
        # the wrapper's structural stream type expects; the ``aclose``/
        # ``close`` overrides below bridge that gap.
        super().__init__(cast(Any, stream))
        self._self_gen = stream
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_output_message: OutputMessage | None = None

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_capture_content:
            return
        output_message = output_message_from_yield_item(chunk)
        if output_message is not None:
            self._self_output_message = output_message

    def _on_stream_end(self) -> None:
        if self._self_output_message is not None:
            self._self_invocation.output_messages = [self._self_output_message]
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)

    async def aclose(self) -> None:
        """Close the underlying async generator and finalize telemetry.

        ``AsyncStreamWrapper.close`` expects the wrapped stream to expose
        ``close``; async generators expose ``aclose`` instead, so mirror the
        base-class behavior here. An early close is treated as a successful
        (partial) completion, matching a caller that stops consuming.
        """
        try:
            await self._self_gen.aclose()
        except Exception as error:
            self._finalize_failure(error)
            raise
        self._finalize_success()

    async def close(self) -> None:
        await self.aclose()


def _build_invocation(
    handler: TelemetryHandler,
    instance: Any,
    msgs: Any,
    request: Any,
) -> AgentInvocation:
    # `agent_name` (config display name with a built-in fallback) was added
    # during the 1.1.x line, so probe for it defensively.
    agent_name = non_empty_str(getattr(instance, "agent_name", None))
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    agent_id = non_empty_str(getattr(instance, "agent_id", None))
    if agent_id is not None:
        invocation.agent_id = agent_id
    conversation_id = non_empty_str(getattr(request, "session_id", None))
    if conversation_id is not None:
        invocation.conversation_id = conversation_id
    if handler.should_capture_content():
        invocation.input_messages = input_messages_from_msgs(msgs)
    return invocation


def make_query_handler_wrapper(
    handler: TelemetryHandler,
) -> Callable[..., Any]:
    """Factory for the ``wrapt`` wrapper bound to *handler*."""

    def query_handler_wrapper(
        wrapped: Callable[..., AsyncGenerator[Any, None]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        msgs, request = parse_query_handler_call(args, kwargs)
        invocation = _build_invocation(handler, instance, msgs, request)
        try:
            stream = wrapped(*args, **kwargs)
        except Exception as exc:
            invocation.fail(exc)
            raise
        return QueryHandlerStreamWrapper(
            stream, invocation, handler.should_capture_content()
        )

    return query_handler_wrapper
