# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Transparent proxy for ``with_raw_response`` / ``with_streaming_response`` results."""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

try:
    import httpx2 as _http_lib
except ImportError:
    import httpx as _http_lib
from anthropic._models import construct_type
from anthropic.types import Message as AnthropicMessage

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

    # wrapt's ``ObjectProxy`` is untyped; model just the constructor and the
    # wrapped-object attribute we use so the proxy subclass is fully typed
    # under pyright.
    class _ObjectProxy:
        __wrapped__: Any

        def __init__(self, wrapped: object) -> None: ...

else:
    from wrapt import ObjectProxy as _ObjectProxy

_logger = logging.getLogger(__name__)

# Distinguishes "a body read owns finalization, stand down" from "no stream
# wrapper to finalize", which are both falsy but call for opposite handling.
_SUPPRESSED = object()


class RawResponseProxy(_ObjectProxy):
    """Proxy for a ``with_raw_response`` / ``with_streaming_response`` result.

    Wrapping the raw response (rather than the parsed payload) keeps every
    metadata attribute -- ``headers``, ``request_id``, ``http_response`` --
    resolving natively and keeps ``isinstance`` / ``__class__`` seeing the
    original type, while ``parse()`` routes on the value it returns:

    * ``Message`` -> extract response telemetry and finalize the span now;
    * ``Stream`` / ``AsyncStream`` -> return an instrumented stream wrapper that
      finalizes the span when drained;
    * a coroutine (the async client's ``parse()``) -> return a coroutine that
      awaits it and dispatches by the same rules;
    * anything else -> return it untouched and log at debug.

    Routing on the parsed value rather than on the request headers is
    deliberate: the SDK sets ``x-stainless-raw-response: stream`` on every
    ``with_streaming_response`` call, streaming or not.

    The span is finalized independently of ``parse()``. A caller that never
    parses either reads the body directly (``read()``, ``.text``, ``.json()``)
    or abandons it, so hooks on the httpx response's ``read`` / ``aread`` and
    ``close`` / ``aclose`` finalize the span when no dispatch did: the read hook
    with response attributes, the close hook with request attributes only.

    Only the instrumented stream wrapper is ever substituted for what the SDK
    returns. Every other ``parse()`` call is delegated, so the caller gets the
    SDK's own object with its cast target, post-parser, and per-target cache
    applied.
    """

    def __init__(
        self,
        raw_response: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(raw_response)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        # The span is ours to end until a stream wrapper takes it over.
        self._self_span_open = True
        # Response telemetry is settled: whichever path saw the body first
        # recorded it (or gave up on it), so no later path may record again.
        self._self_dispatched = False
        # ``parse()`` owns finalization for the body read it drives, and it
        # produces the object the caller should see -- so the read hook stands
        # down and lets the parse dispatch instead.
        self._self_parsing = False
        # A direct body read closes the response before the content is
        # available, so the close hook must stand down until the read returns.
        self._self_reading = False
        # The parsed stream is the one value we hand back in place of the SDK's,
        # so it is memoized; the close hook also drives its finalization when
        # the caller abandons it.
        self._self_stream_wrapper: Any = None
        self._install_hooks(raw_response)

    def _install_hooks(self, raw_response: Any) -> None:
        http_response = getattr(raw_response, "http_response", None)
        if http_response is None:
            return
        # Stacking is safe: each hook calls the one it replaced, so two proxies
        # over one response finalize their own invocation in turn.
        for name, wrap in (
            ("read", self._wrap_sync_read),
            ("aread", self._wrap_async_read),
            ("close", self._wrap_sync_close),
            ("aclose", self._wrap_async_close),
        ):
            original = getattr(http_response, name, None)
            if original is not None:
                setattr(http_response, name, wrap(original))

    def _wrap_sync_read(
        self, original: Callable[..., Any]
    ) -> Callable[..., Any]:
        @functools.wraps(original)
        def _read(*args: object, **kwargs: object) -> Any:
            self._self_reading = True
            try:
                content = original(*args, **kwargs)
            finally:
                self._self_reading = False
            self._finalize_read_fallback()
            return content

        return _read

    def _wrap_async_read(
        self, original: Callable[..., Awaitable[Any]]
    ) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(original)
        async def _aread(*args: object, **kwargs: object) -> Any:
            self._self_reading = True
            try:
                content = await original(*args, **kwargs)
            finally:
                self._self_reading = False
            self._finalize_read_fallback()
            return content

        return _aread

    def _finalize_read_fallback(self) -> None:
        """Finalize a response whose body the caller read without parsing.

        This is the last point at which the deserialized body is available to
        telemetry, so a caller that only ever reads is finalized here with full
        response attributes. A read driven by ``parse()`` is left alone -- the
        dispatch that follows it owns finalization.
        """
        if self._self_parsing or self._self_dispatched:
            return
        message = _message_from_read_body(
            getattr(self.__wrapped__, "http_response", None)
        )
        if message is not None:
            MessageWrapper(message, self._self_capture_content).extract_into(
                self._self_invocation
            )
        # Settled either way: the span ends here, so a later ``parse()`` must
        # not write into an already-stopped invocation.
        self._self_dispatched = True
        self._finalize_once()

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
                await self._afinalize_close_fallback()

        return _aclose

    def _take_stream_wrapper(self) -> Any:
        """The abandoned stream wrapper the close hook has to finalize, if any.

        httpx closes the response while it drains the body and only assigns the
        content afterwards, so a close that fires *during* a read or a
        ``parse()`` would finalize with nothing to extract. Those two own
        finalization themselves once the body is available.

        Handing the wrapper out clears it, which also breaks the recursion from
        finalizing it: closing the wrapper closes the SDK stream, which closes
        this same httpx response and re-enters the hook, where there is now no
        wrapper left and nothing to finalize.
        """
        if self._self_parsing or self._self_reading:
            return _SUPPRESSED
        wrapper = self._self_stream_wrapper
        self._self_stream_wrapper = None
        return wrapper

    def _finalize_close_fallback(self) -> None:
        wrapper = self._take_stream_wrapper()
        if wrapper is _SUPPRESSED:
            return
        if wrapper is None:
            # Nothing read the body, so there are no response attributes to
            # recover; just end the span.
            self._finalize_once()
        elif type(wrapper) is AsyncMessagesStreamWrapper:
            # Nothing here can await the wrapper; httpx also rejects a sync
            # close of an async response, so the ``aclose`` fallback is the one
            # that finalizes this stream.
            self._self_stream_wrapper = wrapper
            _logger.debug(
                "sync close of an async raw-response stream; leaving "
                "finalization to aclose"
            )
        else:
            self._close_stream_wrapper(wrapper)

    async def _afinalize_close_fallback(self) -> None:
        wrapper = self._take_stream_wrapper()
        if wrapper is _SUPPRESSED:
            return
        if wrapper is None:
            # Nothing read the body, so there are no response attributes to
            # recover; just end the span.
            self._finalize_once()
        elif type(wrapper) is AsyncMessagesStreamWrapper:
            try:
                await wrapper.close()
            except Exception:  # pylint: disable=broad-exception-caught
                _logger.debug(
                    "error closing abandoned raw-response stream",
                    exc_info=True,
                )
        else:
            self._close_stream_wrapper(wrapper)

    def _close_stream_wrapper(self, wrapper: Any) -> None:
        try:
            wrapper.close()
        except Exception:  # pylint: disable=broad-exception-caught
            # ``close()`` finalizes the span with the error before re-raising;
            # the caller already abandoned this stream, so don't surface it.
            _logger.debug(
                "error closing abandoned raw-response stream", exc_info=True
            )

    def _finalize_once(self) -> None:
        # ``stop()`` is not idempotent, so finalize at most once.
        if self._self_span_open:
            self._self_span_open = False
            self._self_invocation.stop()

    def parse(self, *args: Any, **kwargs: Any) -> object:
        # The SDK caches per cast target, so every call is delegated and the
        # caller always gets the SDK's own object -- with one exception, the
        # instrumented stream wrapper, which has to be handed back on repeat
        # calls or a second ``parse()`` would return the uninstrumented stream
        # over the same body. A ``parse(to=...)`` call asks for a different cast
        # target, so it is never served from that memo.
        if args or kwargs:
            parsed = self._sdk_parse(*args, **kwargs)
            if inspect.iscoroutine(parsed):
                return self._await_and_settle(parsed, self._settle_cast_parse)
            return self._settle_cast_parse(parsed)
        if self._self_stream_wrapper is not None:
            return self._replay(self._self_stream_wrapper)
        parsed = self._sdk_parse()
        if inspect.iscoroutine(parsed):
            return self._await_and_settle(parsed, self._dispatch)
        return self._dispatch(parsed)

    def _settle_cast_parse(self, parsed: Any) -> object:
        """Record telemetry for a ``parse(to=...)`` and hand back its result.

        A cast parse returns the caller's own target, so there is nothing to
        route on -- but it did read the body, and it suppressed the read hook
        that would otherwise have recorded it. Replay that hook here. A cast
        target the SDK serves without reading (a caller's own ``Stream``
        subclass over an SSE body) leaves nothing to extract, so the read and
        close hooks stay armed for it instead.
        """
        if _body_was_read(getattr(self.__wrapped__, "http_response", None)):
            self._finalize_read_fallback()
        return parsed

    def _sdk_parse(self, *args: Any, **kwargs: Any) -> Any:
        # Suppress the read and close fallbacks only across an actual body read.
        # The sync ``parse()`` reads here, so guard this call; the async
        # ``parse()`` only builds a coroutine and reads nothing until awaited,
        # so ``_await_and_settle`` re-arms the flag around the ``await``
        # instead. Setting it outside a body-read window would strand it True if
        # the caller never awaited the coroutine, leaking the span.
        self._self_parsing = True
        try:
            return self.__wrapped__.parse(*args, **kwargs)
        except Exception:
            self._finalize_failed_parse()
            raise
        finally:
            self._self_parsing = False

    def _finalize_failed_parse(self) -> None:
        # The read that failed may already have closed the body, and the close
        # hook stood down while it ran, so nothing else will end this span.
        # Deserialization is the caller's own failure, not the API call's, so
        # the span stays a success -- just without response attributes.
        self._self_dispatched = True
        self._finalize_once()

    def _replay(self, value: object) -> object:
        if inspect.iscoroutinefunction(
            getattr(self.__wrapped__, "parse", None)
        ):
            return self._replay_async(value)
        return value

    async def _replay_async(self, value: object) -> object:
        return value

    async def _await_and_settle(
        self,
        awaitable: Awaitable[Any],
        settle: Callable[[Any], object],
    ) -> object:
        self._self_parsing = True
        try:
            parsed = await awaitable
        except Exception:
            self._finalize_failed_parse()
            raise
        finally:
            self._self_parsing = False
        return settle(parsed)

    def _dispatch(self, parsed: Any) -> object:
        if self._self_dispatched:
            # A read fallback already settled and ended this span; the caller
            # still gets the SDK's object, just without a second recording.
            return parsed
        if isinstance(parsed, AnthropicMessage):
            MessageWrapper(parsed, self._self_capture_content).extract_into(
                self._self_invocation
            )
            self._self_dispatched = True
            self._finalize_once()
            return parsed
        try:
            wrapped = _wrap_parsed_stream(
                parsed, self._self_invocation, self._self_capture_content
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Same rule as message extraction: a wrapper we failed to build
            # must not turn the caller's successful parse() into an error.
            _logger.debug(
                "failed to wrap a parsed raw-response stream; "
                "returning it uninstrumented",
                exc_info=True,
            )
            wrapped = None
        if wrapped is not None:
            self._self_stream_wrapper = wrapped
            self._self_dispatched = True
            # The wrapper finalizes the span when it is drained or closed.
            self._self_span_open = False
            return wrapped
        # Neither an SDK message nor a stream we can drive; hand it back
        # untouched. The body was likely already read/closed during parse, so
        # finalize now with the request attributes.
        _logger.debug(
            "with_raw_response.parse() returned %s, which we cannot "
            "instrument; skipping response telemetry for this call",
            type(parsed).__name__,
        )
        self._self_dispatched = True
        self._finalize_once()
        return parsed


def _wrap_parsed_stream(
    stream: Any,
    invocation: InferenceInvocation,
    capture_content: bool,
) -> object | None:
    """Wrap a parsed stream in the matching instrumented wrapper.

    The async client's legacy raw response returns an ``AsyncStream`` from its
    *sync* ``parse()``, so both stream types can reach either client's proxy.
    Returns ``None`` for anything we cannot drive.
    """
    if is_anthropic_async_stream(stream):
        return AsyncMessagesStreamWrapper[None](
            cast("AnthropicAsyncStream[RawMessageStreamEvent]", stream),
            invocation,
            capture_content,
        )
    if is_anthropic_stream(stream):
        return MessagesStreamWrapper[None](
            cast("AnthropicStream[RawMessageStreamEvent]", stream),
            invocation,
            capture_content,
        )
    return None


def _body_was_read(http_response: Any) -> bool:
    """Whether the body is in memory; ``content`` raises until it has been read."""
    if http_response is None:
        return False
    try:
        http_response.content  # pylint: disable=pointless-statement
    except _http_lib.ResponseNotRead:
        return False
    return True


def _message_from_read_body(http_response: Any) -> AnthropicMessage | None:
    """Deserialize an already-read response body into a ``Message``.

    Used instead of ``result.parse()`` so telemetry never runs the caller's
    deferred parse, which would apply their cast target and post-parser and
    populate the SDK's parse cache. Returns ``None`` when the body is not a
    ``Message``, in which case the span is finalized with request attributes
    only.

    Deserialization goes through the SDK's own ``construct_type``, the same
    non-strict path the caller's ``parse()`` takes. Strict validation would
    reject any field the installed SDK version does not know yet -- a new
    content block type, a new stop reason -- and drop the whole response's
    telemetry for a response the caller parses without complaint.

    The cost is one extra JSON decode plus model construction per raw-response
    call, on top of the caller's own ``parse()``. That is the price of keeping
    the caller's parse cache untouched; it is only paid on the raw-response
    paths, never on a plain ``messages.create``.
    """
    try:
        body: object = http_response.json()
        if isinstance(body, dict):
            fields = cast("dict[str, object]", body)
            if fields.get("type") == "message":
                return cast(
                    AnthropicMessage,
                    construct_type(type_=AnthropicMessage, value=fields),
                )
    except Exception:  # pylint: disable=broad-exception-caught
        _logger.debug(
            "could not deserialize the raw-response body; skipping response "
            "telemetry for this call",
            exc_info=True,
        )
        return None
    _logger.debug(
        "raw-response body is not an anthropic Message; skipping response "
        "telemetry for this call"
    )
    return None


def wrap_raw_response(
    result: Any,
    invocation: InferenceInvocation,
    capture_content: bool,
) -> Any:
    """Wrap a ``with_raw_response`` / ``with_streaming_response`` result.

    When the SDK has already read the body to completion (the non-streaming
    ``with_raw_response`` path), the operation is over and there is nothing to
    defer to, so the span is finalized here and the caller gets its raw response
    untouched.

    While the body is still open the caller owns when it is read, so the result
    is wrapped in a ``RawResponseProxy``.
    """
    http_response = getattr(result, "http_response", None)
    if getattr(http_response, "is_closed", False):
        message = _message_from_read_body(http_response)
        if message is not None:
            MessageWrapper(message, capture_content).extract_into(invocation)
        invocation.stop()
        return result

    return RawResponseProxy(result, invocation, capture_content)
