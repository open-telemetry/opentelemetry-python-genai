# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrap ``AgentRunner.query_handler`` with an ``invoke_agent`` invocation.

``query_handler`` is an async generator: the invocation must stay open until
the caller drains (or closes) the stream, so the returned generator is
proxied through :class:`QueryHandlerStreamWrapper` which finalizes the
telemetry exactly once on success, error, or close.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from types import TracebackType
from typing import Any, Literal, cast

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

_logger = logging.getLogger(__name__)


class QueryHandlerStreamWrapper(AsyncStreamWrapper[object]):
    """Proxy the ``query_handler`` async generator and finalize telemetry.

    ``AsyncStreamWrapper`` closes the wrapped stream through ``close()``,
    which async generators do not have — they expose ``aclose()`` instead.
    The close overrides below bridge that gap so the generator is always
    closed and telemetry is finalized exactly once.
    """

    def __init__(
        self,
        stream: AsyncGenerator[object, None],
        invocation: AgentInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(cast(Any, stream))
        self._self_gen = stream
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_output_message: OutputMessage | None = None

    def _process_chunk(self, chunk: object) -> None:
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

        An early close is treated as a successful (partial) completion,
        matching a caller that stops consuming.
        """
        try:
            await self._self_gen.aclose()
        except Exception as error:
            self._finalize_failure(error)
            raise
        self._finalize_success()

    async def close(self) -> None:
        await self.aclose()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_val is None:
            return await super().__aexit__(exc_type, exc_val, exc_tb)
        # The base class closes the stream via ``close()`` here, so close the
        # generator ourselves; failing to do so would leave it running.
        self._finalize_failure(exc_val)
        try:
            await self._self_gen.aclose()
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "QwenPaw stream close error after user exception",
                exc_info=True,
            )
        return False


def _build_invocation(
    handler: TelemetryHandler,
    instance: object,
    msgs: object,
    request: object,
) -> AgentInvocation:
    # `agent_name` (config display name with a built-in fallback) was added
    # during the 1.1.x line, so probe for it defensively.
    agent_name = non_empty_str(getattr(instance, "agent_name", None))
    invocation = handler.invoke_local_agent(agent_name=agent_name)
    # The runner's `agent_id` is a local config key (e.g. "default"), not a
    # provider-assigned stable identifier, so `gen_ai.agent.id` is not
    # recorded per its semconv guidance.
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
        wrapped: Callable[..., AsyncGenerator[object, None]],
        # The runner is only attribute-probed via ``getattr`` (``agent_name``
        # arrived during the 1.1.x line), and ``qwenpaw`` cannot be imported
        # for typing on Python >= 3.14, so ``object`` is the narrowest
        # honest annotation here.
        instance: object,
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
