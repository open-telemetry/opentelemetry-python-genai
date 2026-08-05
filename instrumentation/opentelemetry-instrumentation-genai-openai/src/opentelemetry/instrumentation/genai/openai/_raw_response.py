# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Transparent proxy for streaming ``with_raw_response`` results."""

from __future__ import annotations

import functools
import logging
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    Union,
    runtime_checkable,
)

import httpx
from openai import AsyncStream, Stream
from wrapt import ObjectProxy

if TYPE_CHECKING:
    from opentelemetry.util.genai.types import GenAIInvocation

_logger = logging.getLogger(__name__)

AnyStream = Union[Stream[Any], AsyncStream[Any]]


@runtime_checkable
class ParsableResponse(Protocol):
    """A ``with_raw_response`` result whose payload is behind ``parse()``.

    Structural rather than nominal: the SDK's response classes
    (``LegacyAPIResponse``, ``APIResponse``, ``AsyncAPIResponse``) all match,
    but they live in private modules, so matching on shape keeps us working
    across SDK versions without importing private names.
    """

    def parse(self, *args: Any, **kwargs: Any) -> Any: ...


@runtime_checkable
class RawResponseLike(ParsableResponse, Protocol):
    """A parsable response we can also drive to completion.

    Adds the underlying httpx response, which the streaming path needs to
    finalize the span when the caller never calls ``parse()``.
    """

    http_response: httpx.Response


class RawResponseStreamProxy(ObjectProxy):
    """Proxy for a streaming ``with_raw_response`` result.

    The OpenAI SDK returns a raw-response object (a ``LegacyAPIResponse``) from
    ``with_raw_response.create(stream=True)``; callers read response metadata
    (``headers``, ``request_id`` ...) off it and call ``parse()`` to obtain the
    stream. Wrapping that response (instead of the parsed stream) keeps every
    metadata attribute resolving natively while ``parse()`` returns an
    instrumented stream wrapper. Parsing is deferred until the caller asks for
    it and memoized so repeated calls share one span.

    ``parse()`` wraps the result only when it is an SDK ``Stream`` /
    ``AsyncStream`` we know how to drive. Anything else is handed back untouched
    (for example the coroutine ``parse()`` the SDK documents it "will become in
    the next major version" for the async client, or a custom non-stream parse
    target): we can't instrument it, so we just log and step aside.

    The span is finalized independently of ``parse()``. Whether the caller
    parses and drains the wrapper, drains the body directly via the raw
    response's ``read()``/``iter_bytes()``, or never parses at all, every path
    ends by closing the underlying httpx response — so a fallback on its
    ``close``/``aclose`` finalizes the span when ``parse()`` never built a
    wrapper that would finalize it instead.
    """

    def __init__(
        self,
        raw_response: RawResponseLike,
        wrap_stream: Callable[[AnyStream], object],
        finalize: Callable[[], None],
    ) -> None:
        super().__init__(raw_response)
        self._self_wrap_stream = wrap_stream
        self._self_finalize: Callable[[], None] | None = finalize
        self._self_parsed: object | None = None
        self._install_close_fallback(raw_response)

    def _install_close_fallback(self, raw_response: RawResponseLike) -> None:
        # httpx exposes a sync ``close`` and an async ``aclose``; wrap whichever
        # the underlying response has so the appropriate one finalizes the span.
        http_response = raw_response.http_response
        close = getattr(http_response, "close", None)
        if close is not None:
            http_response.close = self._wrap_sync_close(close)
        aclose = getattr(http_response, "aclose", None)
        if aclose is not None:
            http_response.aclose = self._wrap_async_close(aclose)

    def _wrap_sync_close(
        self, original: Callable[..., object]
    ) -> Callable[..., object]:
        @functools.wraps(original)
        def _close(*args: object, **kwargs: object) -> object:
            try:
                return original(*args, **kwargs)
            finally:
                self._finalize_close_fallback()

        return _close

    def _wrap_async_close(
        self, original: Callable[..., Awaitable[object]]
    ) -> Callable[..., Awaitable[object]]:
        @functools.wraps(original)
        async def _aclose(*args: object, **kwargs: object) -> object:
            try:
                return await original(*args, **kwargs)
            finally:
                self._finalize_close_fallback()

        return _aclose

    def _finalize_close_fallback(self) -> None:
        # When ``parse()`` built a stream wrapper, that wrapper owns
        # finalization; only finalize here for callers that never parsed.
        if self._self_parsed is None:
            self._finalize_once()

    def _finalize_once(self) -> None:
        # ``stop()`` is not idempotent, so finalize at most once.
        if self._self_finalize is not None:
            finalize, self._self_finalize = self._self_finalize, None
            finalize()

    def parse(self, *args: Any, **kwargs: Any) -> object:
        # We memoize the first parse regardless of the arguments it was called
        # with, so calling parse() again with different args (e.g. a different
        # ``to=`` type) returns the same wrapper rather than re-parsing. This is
        # deliberate: one raw response backs one span, and re-parsing a stream
        # would consume it twice anyway.
        if self._self_parsed is not None:
            return self._self_parsed
        stream = self.__wrapped__.parse(*args, **kwargs)
        if isinstance(stream, (Stream, AsyncStream)):
            self._self_parsed = self._self_wrap_stream(stream)
            return self._self_parsed
        # Not a stream we can drive; hand it back untouched. The close fallback
        # finalizes the span once the caller drains/closes the response body.
        _logger.debug(
            "with_raw_response.parse() returned %s, not an SDK stream; "
            "skipping stream instrumentation for this call",
            type(stream).__name__,
        )
        return stream


class StreamWrapperFactory(Protocol):
    """Constructor signature every stream wrapper class must satisfy."""

    def __call__(
        self,
        stream: AnyStream,
        invocation: GenAIInvocation,
        capture_content: bool,
    ) -> object: ...


def wrap_stream_result(
    wrapper_cls: StreamWrapperFactory,
    result: RawResponseLike | AnyStream,
    invocation: GenAIInvocation,
    capture_content: bool,
) -> object:
    """Wrap a streaming call's result, deferring ``parse()`` on raw responses.

    A ``with_raw_response`` result (matching ``RawResponseLike``) is wrapped so
    its metadata resolves natively and ``parse()`` is deferred; a plain SDK
    stream is wrapped directly.
    """
    if isinstance(result, RawResponseLike):
        return RawResponseStreamProxy(
            result,
            lambda stream: wrapper_cls(stream, invocation, capture_content),
            finalize=invocation.stop,
        )
    return wrapper_cls(result, invocation, capture_content)
