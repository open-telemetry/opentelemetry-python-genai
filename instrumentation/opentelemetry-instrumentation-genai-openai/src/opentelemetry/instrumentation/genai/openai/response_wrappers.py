# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for OpenAI Responses API streams and stream managers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextvars import ContextVar
from types import TracebackType
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from opentelemetry.util.genai.stream import (
    AsyncStreamManagerWrapper,
    AsyncStreamWrapper,
    SyncStreamManagerWrapper,
    SyncStreamWrapper,
    finalize_on_aclose,
    finalize_on_close,
)
from opentelemetry.util.genai.types import Error

try:
    from opentelemetry.instrumentation.genai.openai.response_extractors import (  # pylint: disable=no-name-in-module
        get_response_error,
        set_fetch_response_attributes,
        set_invocation_response_attributes,
    )
except ImportError:
    get_response_error = None
    set_fetch_response_attributes = None
    set_invocation_response_attributes = None

try:
    from opentelemetry.instrumentation.genai.openai.utils import (  # pylint: disable=no-name-in-module
        get_served_model,
    )
except ImportError:
    get_served_model = None

if TYPE_CHECKING:
    from openai.lib.streaming.responses._events import (  # pylint: disable=no-name-in-module
        ResponseStreamEvent,
    )
    from openai.lib.streaming.responses._responses import (
        AsyncResponseStream,
        AsyncResponseStreamManager,
        ResponseStream,
        ResponseStreamManager,
    )  # pylint: disable=no-name-in-module
    from openai.types.responses import (  # pylint: disable=no-name-in-module
        ParsedResponse,
        Response,
    )

    from opentelemetry.util.genai._invocation import GenAIInvocation

_logger = logging.getLogger(__name__)

TextFormatT = TypeVar("TextFormatT")
ResponseT = TypeVar("ResponseT")
ResponsesStreamContext = tuple["GenAIInvocation", bool]

responses_stream_context: ContextVar[ResponsesStreamContext | None] = (
    ContextVar("responses_stream_context", default=None)
)


def _set_responses_stream_context(
    invocation: GenAIInvocation, capture_content: bool
):
    """Mark the current ``Responses.stream`` manager entry.

    OpenAI's ``Responses.stream`` manager enters by calling the patched
    ``Responses.create(..., stream=True)`` method internally. This context
    marker lets the inner ``create`` wrapper return the SDK stream without
    creating a second telemetry invocation; the outer stream manager wrapper
    owns the span lifecycle.

    :returns: The ``ContextVar`` reset handle for restoring the previous
        context after the SDK manager has entered.
    """
    return responses_stream_context.set((invocation, capture_content))


def _set_response_attributes(
    invocation: GenAIInvocation,
    result: ParsedResponse[TextFormatT] | Response | None,
    capture_content: bool,
) -> None:
    if set_invocation_response_attributes is None:
        return
    set_invocation_response_attributes(invocation, result, capture_content)


def _set_fetch_response_attributes(
    invocation: GenAIInvocation,
    result: ParsedResponse[TextFormatT] | Response | None,
    capture_content: bool,
) -> None:
    if set_fetch_response_attributes is None:
        return
    set_fetch_response_attributes(invocation, result, capture_content)


def _get_stream_response(stream):
    try:
        return stream._response
    except AttributeError:
        try:
            return stream.response
        except AttributeError:
            return None


class _ResponseStreamMixin(Generic[TextFormatT]):
    _self_invocation: GenAIInvocation
    _self_capture_content: bool
    _self_response_telemetry_finalized: bool

    def __init__(
        self,
        invocation: GenAIInvocation,
        capture_content: bool,
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_response_telemetry_finalized = False

    def _stop(
        self, result: ParsedResponse[TextFormatT] | Response | None
    ) -> None:
        if self._self_response_telemetry_finalized:
            return
        self._apply_response_attributes(result)
        if get_served_model is not None:
            stream = getattr(self, "stream", None)
            if stream is not None:
                underlying = _get_stream_response(stream)
                served_model = get_served_model(
                    getattr(underlying, "headers", None)
                )
                if served_model:
                    self._self_invocation.response_model_name = served_model
        self._self_invocation.stop()
        self._self_response_telemetry_finalized = True

    def _apply_response_attributes(
        self, result: ParsedResponse[TextFormatT] | Response | None
    ) -> None:
        """Record the response on the invocation. Overridden per operation."""
        _set_response_attributes(
            self._self_invocation, result, self._self_capture_content
        )

    def _on_response_failed(
        self, response: ParsedResponse[TextFormatT] | Response | None
    ) -> None:
        """Handle a ``response.failed`` event. Overridden per operation."""
        self._apply_response_attributes(response)
        error = get_response_error(response) if get_response_error else None
        self._fail(error or Error(type="response.failed", message=None))

    def _fail(self, error: Error) -> None:
        if self._self_response_telemetry_finalized:
            return
        self._self_invocation.fail(error)
        self._self_response_telemetry_finalized = True

    def _process_chunk(self, chunk: ResponseStreamEvent[TextFormatT]) -> None:
        self.process_event(chunk)

    def _on_stream_end(self) -> None:
        self._stop(None)

    def _on_stream_error(self, error: BaseException) -> None:
        if self._self_response_telemetry_finalized:
            return
        self._self_invocation.fail(error)
        self._self_response_telemetry_finalized = True

    def get_final_response(self) -> ParsedResponse[TextFormatT]:
        self.until_done()
        return self.stream.get_final_response()

    def until_done(self) -> ResponseStreamWrapper:
        for _ in self:
            pass
        return self

    @property
    def response(self):
        response = _get_stream_response(self.stream)
        if response is None:
            return None
        return finalize_on_close(response, lambda: self._stop(None))

    def process_event(self, event: ResponseStreamEvent[TextFormatT]) -> None:
        # raw-response stream can be parsed into a caller-defined event type.
        event_type = getattr(event, "type", None)
        if not isinstance(event_type, str):
            _logger.debug(
                "Skipping telemetry for unrecognized response event type %s",
                type(event).__name__,
            )
            return

        response: ParsedResponse[TextFormatT] | Response | None = getattr(
            event, "response", None
        )

        if response and not self._self_invocation.response_model_name:
            model = getattr(response, "model", None)
            if model:
                self._self_invocation.response_model_name = model

        if event_type in {"response.completed", "response.incomplete"}:
            self._stop(response)
            return

        if event_type == "response.failed":
            self._on_response_failed(response)
            return

        if event.type == "error":
            error_type = event.code or "error"
            message = event.message or None
            self._fail(Error(type=error_type, message=message))


class ResponseStreamWrapper(
    _ResponseStreamMixin[TextFormatT],
    SyncStreamWrapper["ResponseStreamEvent[TextFormatT]"],
    Generic[TextFormatT],
):
    """Wrapper for OpenAI Responses API stream objects.

    Wraps ResponseStream from the OpenAI SDK:
    https://github.com/openai/openai-python/blob/656e3cab4a18262a49b961d41293367e45ee71b9/src/openai/_streaming.py#L55
    """

    def __init__(
        self,
        stream: ResponseStream[TextFormatT],
        invocation: GenAIInvocation,
        capture_content: bool,
    ):
        SyncStreamWrapper.__init__(self, stream, invocation=invocation)
        # Marks streams already wrapped by the inner Responses.create path so
        # Responses.stream manager entry can return them without wrapping twice.
        self._self_is_response_stream_wrapper = True
        _ResponseStreamMixin.__init__(self, invocation, capture_content)

    @property
    def stream(self) -> ResponseStream[TextFormatT]:
        return self._self_stream

    @stream.setter
    def stream(self, stream: ResponseStream[TextFormatT]) -> None:
        self._set_stream(stream)


class _FetchResponseStreamMixin(Generic[TextFormatT]):
    """Finalization overrides for a streamed ``responses.retrieve``.

    The stream replays a response generated by an earlier operation, so the
    fetch-response attributes are recorded instead of the inference ones, and a
    replayed ``response.failed`` describes that original generation rather than
    a failure of this fetch.
    """

    _self_invocation: GenAIInvocation
    _self_capture_content: bool

    def _apply_response_attributes(
        self, result: ParsedResponse[TextFormatT] | Response | None
    ) -> None:
        _set_fetch_response_attributes(
            self._self_invocation, result, self._self_capture_content
        )

    def _on_response_failed(
        self, response: ParsedResponse[TextFormatT] | Response | None
    ) -> None:
        self._stop(response)


class FetchResponseStreamWrapper(
    _FetchResponseStreamMixin[TextFormatT],
    ResponseStreamWrapper[TextFormatT],
    Generic[TextFormatT],
):
    """Wrapper for a streamed ``Responses.retrieve`` replay."""


class ResponseStreamManagerWrapper(
    SyncStreamManagerWrapper[
        "ResponseStream[TextFormatT]",
        "GenAIInvocation",
        "ResponseStreamWrapper[TextFormatT]",
    ],
    Generic[TextFormatT],
):
    """Wrapper for OpenAI Responses API stream managers.

    Wraps ResponseStreamManager from the OpenAI SDK:
    https://github.com/openai/openai-python/blob/656e3cab4a18262a49b961d41293367e45ee71b9/src/openai/lib/streaming/responses/_responses.py#L95
    """

    def __init__(
        self,
        manager: ResponseStreamManager[TextFormatT],
        invocation_factory: Callable[[], GenAIInvocation],
        capture_content: bool,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content

    def _enter_manager(
        self, invocation: GenAIInvocation
    ) -> ResponseStream[TextFormatT]:
        # The SDK manager enters by calling the patched Responses.create, which
        # this marker tells to return the SDK stream without opening a second
        # invocation.
        stream_context_reset = _set_responses_stream_context(
            invocation, self._self_capture_content
        )
        try:
            return cast(
                "ResponseStream[TextFormatT]", self.__wrapped__.__enter__()
            )
        finally:
            responses_stream_context.reset(stream_context_reset)

    def _wrap_stream(
        self, stream: ResponseStream[TextFormatT], invocation: GenAIInvocation
    ) -> ResponseStreamWrapper[TextFormatT]:
        if getattr(stream, "_self_is_response_stream_wrapper", False):
            # Already wrapped by the inner patched create().
            return cast("ResponseStreamWrapper[TextFormatT]", stream)
        return ResponseStreamWrapper(
            stream, invocation, self._self_capture_content
        )

    def parse(self) -> ResponseStreamManagerWrapper[TextFormatT]:
        raise NotImplementedError(
            "ResponseStreamManagerWrapper.parse() is not implemented"
        )


class AsyncResponseStreamWrapper(
    _ResponseStreamMixin[TextFormatT],
    AsyncStreamWrapper["ResponseStreamEvent[TextFormatT]"],
    Generic[TextFormatT],
):
    """Wrapper for async OpenAI Responses API stream objects."""

    stream: AsyncResponseStream[TextFormatT]

    def __init__(
        self,
        stream: AsyncResponseStream[TextFormatT],
        invocation: GenAIInvocation,
        capture_content: bool,
    ):
        AsyncStreamWrapper.__init__(self, stream, invocation=invocation)
        # Marks streams already wrapped by the inner AsyncResponses.create path
        # so AsyncResponses.stream manager entry avoids wrapping twice.
        self._self_is_response_stream_wrapper = True
        _ResponseStreamMixin.__init__(self, invocation, capture_content)

    async def __aenter__(self) -> AsyncResponseStreamWrapper[TextFormatT]:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return await AsyncStreamWrapper.__aexit__(
            self, exc_type, exc_val, exc_tb
        )

    async def get_final_response(self) -> ParsedResponse[TextFormatT]:
        await self.until_done()
        return await self.stream.get_final_response()

    async def until_done(self) -> AsyncResponseStreamWrapper[TextFormatT]:
        async for _ in self:
            pass
        return self

    @property
    def stream(self) -> AsyncResponseStream[TextFormatT]:
        return self._self_stream

    @stream.setter
    def stream(self, stream: AsyncResponseStream[TextFormatT]) -> None:
        self._set_stream(stream)

    @property
    def response(self):
        response = _get_stream_response(self.stream)
        if response is None:
            return None
        return finalize_on_aclose(response, lambda: self._stop(None))


class AsyncFetchResponseStreamWrapper(
    _FetchResponseStreamMixin[TextFormatT],
    AsyncResponseStreamWrapper[TextFormatT],
    Generic[TextFormatT],
):
    """Wrapper for a streamed ``AsyncResponses.retrieve`` replay."""


class AsyncResponseStreamManagerWrapper(
    AsyncStreamManagerWrapper[
        "AsyncResponseStream[TextFormatT]",
        "GenAIInvocation",
        "AsyncResponseStreamWrapper[TextFormatT]",
    ],
    Generic[TextFormatT],
):
    """Wrapper for async OpenAI Responses API stream managers."""

    def __init__(
        self,
        manager: AsyncResponseStreamManager[TextFormatT],
        invocation_factory: Callable[[], GenAIInvocation],
        capture_content: bool,
    ):
        super().__init__(manager, invocation_factory)
        self._self_capture_content = capture_content

    async def _enter_manager(
        self, invocation: GenAIInvocation
    ) -> AsyncResponseStream[TextFormatT]:
        # See ResponseStreamManagerWrapper._enter_manager.
        stream_context_reset = _set_responses_stream_context(
            invocation, self._self_capture_content
        )
        try:
            return cast(
                "AsyncResponseStream[TextFormatT]",
                await self.__wrapped__.__aenter__(),
            )
        finally:
            responses_stream_context.reset(stream_context_reset)

    def _wrap_stream(
        self,
        stream: AsyncResponseStream[TextFormatT],
        invocation: GenAIInvocation,
    ) -> AsyncResponseStreamWrapper[TextFormatT]:
        if getattr(stream, "_self_is_response_stream_wrapper", False):
            # Already wrapped by the inner patched create().
            return cast("AsyncResponseStreamWrapper[TextFormatT]", stream)
        return AsyncResponseStreamWrapper(
            stream, invocation, self._self_capture_content
        )

    def parse(self) -> AsyncResponseStreamManagerWrapper[TextFormatT]:
        raise NotImplementedError(
            "AsyncResponseStreamManagerWrapper.parse() is not implemented"
        )
