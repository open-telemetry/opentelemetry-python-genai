# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Anthropic hooks for the shared ``with_raw_response`` proxy."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from anthropic._models import construct_type
from anthropic.types import Message as AnthropicMessage

from opentelemetry.util.genai.raw_response import (
    RawResponseProxy,
)
from opentelemetry.util.genai.raw_response import (
    wrap_raw_response as _wrap_raw_response,
)

from .utils import is_anthropic_async_stream, is_anthropic_stream
from .wrappers import (
    AsyncMessagesStreamWrapper,
    MessagesStreamWrapper,
    MessageWrapper,
)

if TYPE_CHECKING:
    from anthropic._streaming import AsyncStream as AnthropicAsyncStream
    from anthropic._streaming import Stream as AnthropicStream
    from anthropic.types import RawMessageStreamEvent

    from opentelemetry.util.genai.invocation import InferenceInvocation

_logger = logging.getLogger(__name__)


class MessagesRawResponseProxy(RawResponseProxy):
    """Routes a raw ``messages.create`` response to Anthropic telemetry."""

    def __init__(
        self,
        raw_response: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(raw_response, invocation)
        self._self_capture_content = capture_content

    def _extract_from_body(self, body: dict[str, object]) -> bool:
        message = _message_from_body(body)
        if message is None:
            return False
        self._extract_message(message)
        return True

    def _extract_from_parsed(self, parsed: Any) -> bool:
        if not isinstance(parsed, AnthropicMessage):
            return False
        self._extract_message(parsed)
        return True

    def _extract_message(self, message: AnthropicMessage) -> None:
        MessageWrapper(message, self._self_capture_content).extract_into(
            cast("InferenceInvocation", self._self_invocation)
        )

    def _wrap_parsed_stream(self, parsed: Any) -> object | None:
        invocation = cast("InferenceInvocation", self._self_invocation)
        if is_anthropic_async_stream(parsed):
            return AsyncMessagesStreamWrapper[None](
                cast("AnthropicAsyncStream[RawMessageStreamEvent]", parsed),
                invocation,
                self._self_capture_content,
            )
        if is_anthropic_stream(parsed):
            return MessagesStreamWrapper[None](
                cast("AnthropicStream[RawMessageStreamEvent]", parsed),
                invocation,
                self._self_capture_content,
            )
        return None


def _message_from_body(body: dict[str, object]) -> AnthropicMessage | None:
    """Construct a ``Message`` from an already-decoded response body.

    Returns ``None`` when the body is not a ``Message``, in which case the span
    is finalized with request attributes only.

    Deserialization goes through the SDK's own ``construct_type``, the same
    non-strict path the caller's ``parse()`` takes. Strict validation would
    reject any field the installed SDK version does not know yet -- a new
    content block type, a new stop reason -- and drop the whole response's
    telemetry for a response the caller parses without complaint.

    The cost is one extra model construction per raw-response call, on top of
    the caller's own ``parse()``. That is the price of keeping the caller's
    parse cache untouched; it is only paid on the raw-response paths, never on
    a plain ``messages.create``.
    """
    if body.get("type") != "message":
        _logger.debug(
            "raw-response body is not an anthropic Message; skipping response "
            "telemetry for this call"
        )
        return None
    try:
        return cast(
            AnthropicMessage,
            construct_type(type_=AnthropicMessage, value=body),
        )
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.debug(
            "could not construct the raw-response Message; skipping response "
            "telemetry for this call",
            exc_info=True,
        )
        return None


def wrap_raw_response(
    result: Any,
    invocation: InferenceInvocation,
    capture_content: bool,
    streamed: bool = False,
) -> Any:
    """Wrap a ``with_raw_response`` / ``with_streaming_response`` result."""
    return _wrap_raw_response(
        lambda raw: MessagesRawResponseProxy(raw, invocation, capture_content),
        result,
        streamed=streamed,
    )
