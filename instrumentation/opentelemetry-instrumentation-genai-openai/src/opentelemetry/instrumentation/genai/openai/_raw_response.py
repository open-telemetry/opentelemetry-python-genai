# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Transparent proxy for streaming ``with_raw_response`` results."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Callable

from wrapt import ObjectProxy

if TYPE_CHECKING:
    from opentelemetry.util.genai.types import GenAIInvocation

_logger = logging.getLogger(__name__)


class RawResponseStreamProxy(ObjectProxy):
    """Proxy for a streaming ``with_raw_response`` result.

    The OpenAI SDK returns a raw-response object (a ``LegacyAPIResponse``) from
    ``with_raw_response.create(stream=True)``; callers read response metadata
    (``headers``, ``request_id`` ...) off it and call ``parse()`` to obtain the
    stream. Wrapping that response (instead of the parsed stream) keeps every
    metadata attribute resolving natively while ``parse()`` returns an
    instrumented stream wrapper. Parsing is deferred until the caller asks for
    it and memoized so repeated calls share one span.

    ``LegacyAPIResponse.parse()`` returns the stream synchronously today, but
    the SDK documents that it "will become a coroutine in the next major
    version" for the async client. If it ever hands back something other than a
    synchronous stream (e.g. an awaitable), we can't wrap it without breaking
    the caller, so we back off: finalize the span (nothing will drive the
    stream wrapper to close it) and return the SDK result untouched. Telemetry
    is degraded for that call, but the caller's code keeps working unchanged.
    """

    def __init__(
        self,
        raw_response: object,
        wrap_stream: Callable[[object], object],
        on_backoff: Callable[[], None],
    ) -> None:
        super().__init__(raw_response)
        self._self_wrap_stream = wrap_stream
        self._self_on_backoff: Callable[[], None] | None = on_backoff
        self._self_parsed: object | None = None

    def parse(self, *args: object, **kwargs: object) -> object:
        if self._self_parsed is None:
            stream = self.__wrapped__.parse(*args, **kwargs)
            if inspect.isawaitable(stream):
                _logger.debug(
                    "with_raw_response.parse() returned %s, not a synchronous "
                    "stream; skipping stream instrumentation for this call",
                    type(stream).__name__,
                )
                self._finalize_on_backoff()
                return stream
            self._self_parsed = self._self_wrap_stream(stream)
        return self._self_parsed

    def _finalize_on_backoff(self) -> None:
        # ``stop()`` is not idempotent, so finalize at most once.
        if self._self_on_backoff is not None:
            on_backoff, self._self_on_backoff = self._self_on_backoff, None
            on_backoff()


class StreamResultFactory:
    """Mixin adding ``wrap_result`` to a stream wrapper class.

    The concrete wrapper must accept ``(stream, invocation, capture_content)``.
    Call it from a streaming branch: a ``with_raw_response`` result (a
    ``LegacyAPIResponse``, detected by its ``parse`` method) is wrapped so its
    metadata resolves natively and ``parse()`` is deferred; a plain SDK stream
    is wrapped directly.
    """

    @classmethod
    def wrap_result(
        cls: type,
        result: object,
        invocation: GenAIInvocation,
        capture_content: bool,
    ) -> object:
        if hasattr(result, "parse"):
            return RawResponseStreamProxy(
                result,
                lambda stream: cls(stream, invocation, capture_content),
                on_backoff=invocation.stop,
            )
        return cls(result, invocation, capture_content)
