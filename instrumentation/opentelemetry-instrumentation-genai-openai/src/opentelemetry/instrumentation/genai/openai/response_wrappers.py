# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Wrappers for OpenAI Responses API streams and stream managers."""

from __future__ import annotations

from contextlib import AsyncExitStack, ExitStack
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import Error

try:
    from opentelemetry.instrumentation.genai.openai.response_extractors import (  # pylint: disable=no-name-in-module
        set_invocation_response_attributes,
    )
except ImportError:
    set_invocation_response_attributes = None

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

TextFormatT = TypeVar("TextFormatT")
ResponseT = TypeVar("ResponseT")


def _set_response_attributes(
    invocation: "GenAIInvocation",
    result: "ParsedResponse[TextFormatT] | Response | None",
    capture_content: bool,
) -> None:
    if set_invocation_response_attributes is None:
        return
    set_invocation_response_attributes(invocation, result, capture_content)


def _get_stream_response(stream):
    try:
        return stream._response
    except AttributeError:
        try:
            return stream.response
        except AttributeError:
            return None


class _ResponseProxy(Generic[ResponseT]):
    def __init__(self, response: ResponseT, finalize: Callable[[], None]):
        self._response = response
        self._finalize = finalize

    def close(self) -> None:
        try:
            self._response.close()
        finally:
            self._finalize()

    def __getattr__(self, name: str):
        return getattr(self._response, name)


class _AsyncResponseProxy(Generic[ResponseT]):
    def __init__(self, response: ResponseT, finalize: Callable[[], None]):
        self._response = response
        self._finalize = finalize

    async def aclose(self) -> None:
        try:
            await self._response.aclose()
        finally:
            self._finalize()

    def __getattr__(self, name: str):
        return getattr(self._response, name)


class _ResponseStreamMixin(Generic[TextFormatT]):
    _self_invocation: "GenAIInvocation"
    _self_capture_content: bool
    _self_response_telemetry_finalized: bool

    def __init__(
        self,
        invocation: "GenAIInvocation",
        capture_content: bool,
    ) -> None:
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_response_telemetry_finalized = False

    def _stop(
        self, result: "ParsedResponse[TextFormatT] | Response | None"
    ) -> None:
        if self._self_response_telemetry_finalized:
            return
        _set_response_attributes(
            self._self_invocation, result, self._self_capture_content
        )
        self._self_invocation.stop()
        self._self_response_telemetry_finalized = True

    def _fail(self, message: str, error_type: type[BaseException]) -> None:
        if self._self_response_telemetry_finalized:
            return
        self._self_invocation.fail(Error(message=message, type=error_type))
        self._self_response_telemetry_finalized = True

    def _process_chunk(
        self, chunk: "ResponseStreamEvent[TextFormatT]"
    ) -> None:
        self.process_event(chunk)

    def _on_stream_end(self) -> None:
        self._stop(None)

    def _on_stream_error(self, error: BaseException) -> None:
        self._fail(str(error), type(error))

    def get_final_response(self) -> "ParsedResponse[TextFormatT]":
        self.until_done()
        return self.stream.get_final_response()

    def until_done(self) -> "ResponseStreamWrapper":
        for _ in self:
            pass
        return self

    def parse(self) -> "ResponseStreamWrapper":
        """Called when using with_raw_response with stream=True."""
        return self

    @property
    def response(self):
        response = _get_stream_response(self.stream)
        if response is None:
            return None
        return _ResponseProxy(response, lambda: self._stop(None))

    def process_event(self, event: "ResponseStreamEvent[TextFormatT]") -> None:
        event_type = event.type
        response: "ParsedResponse[TextFormatT] | Response | None" = getattr(
            event, "response", None
        )

        if response and not self._self_invocation.request_model:
            model = response.model
            if model:
                self._self_invocation.request_model = model

        if event_type == "response.completed":
            self._stop(response)
            return

        if event_type in {"response.failed", "response.incomplete"}:
            _set_response_attributes(
                self._self_invocation,
                response,
                self._self_capture_content,
            )
            self._fail(event_type, RuntimeError)
            return

        if event_type == "response.error":
            error_type = getattr(event, "code", None) or "response.error"
            message = getattr(event, "message", None) or error_type
            self._fail(message, RuntimeError)


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
        stream: "ResponseStream[TextFormatT]",
        invocation: "GenAIInvocation",
        capture_content: bool,
    ):
        SyncStreamWrapper.__init__(self, stream)
        _ResponseStreamMixin.__init__(self, invocation, capture_content)

    @property
    def stream(self) -> "ResponseStream[TextFormatT]":
        return self._self_stream

    @stream.setter
    def stream(self, stream: "ResponseStream[TextFormatT]") -> None:
        self.__wrapped__ = stream
        self._self_stream = stream
        self._self_iterator = iter(stream)


class ResponseStreamManagerWrapper(Generic[TextFormatT]):
    """Wrapper for OpenAI Responses API stream managers.

    Wraps ResponseStreamManager from the OpenAI SDK:
    https://github.com/openai/openai-python/blob/656e3cab4a18262a49b961d41293367e45ee71b9/src/openai/lib/streaming/responses/_responses.py#L95
    """

    def __init__(
        self,
        manager: "ResponseStreamManager[TextFormatT]",
        invocation,
        capture_content: bool,
    ):
        self._manager = manager
        self._invocation = invocation
        self._capture_content = capture_content
        self._stream_wrapper: ResponseStreamWrapper[TextFormatT] | None = None

    def __enter__(self) -> ResponseStreamWrapper[TextFormatT]:
        stream = self._manager.__enter__()
        self._stream_wrapper = ResponseStreamWrapper(
            stream,
            self._invocation,
            self._capture_content,
        )
        return self._stream_wrapper

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        suppressed = False
        stream_wrapper = self._stream_wrapper
        self._stream_wrapper = None
        with ExitStack() as cleanup:
            if stream_wrapper is not None:

                def finalize_stream_wrapper() -> None:
                    if suppressed:
                        stream_wrapper.__exit__(None, None, None)
                    else:
                        stream_wrapper.__exit__(exc_type, exc_val, exc_tb)

                cleanup.callback(finalize_stream_wrapper)
            suppressed = self._manager.__exit__(exc_type, exc_val, exc_tb)
            return suppressed

    def parse(self) -> "ResponseStreamManagerWrapper[TextFormatT]":
        raise NotImplementedError(
            "ResponseStreamManagerWrapper.parse() is not implemented"
        )

    # TODO: Replace __getattr__ passthrough with wrapt.ObjectProxy in a future
    # cleanup once wrapt 2 typing support is available (wrapt PR #3903).
    def __getattr__(self, name: str):
        return getattr(self._manager, name)


class AsyncResponseStreamWrapper(
    _ResponseStreamMixin[TextFormatT],
    AsyncStreamWrapper["ResponseStreamEvent[TextFormatT]"],
    Generic[TextFormatT],
):
    """Wrapper for async OpenAI Responses API stream objects."""

    stream: "AsyncResponseStream[TextFormatT]"

    def __init__(
        self,
        stream: "AsyncResponseStream[TextFormatT]",
        invocation: "GenAIInvocation",
        capture_content: bool,
    ):
        AsyncStreamWrapper.__init__(self, stream)
        _ResponseStreamMixin.__init__(self, invocation, capture_content)

    async def __aenter__(self) -> "AsyncResponseStreamWrapper[TextFormatT]":
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

    async def get_final_response(self) -> "ParsedResponse[TextFormatT]":
        await self.until_done()
        return await self.stream.get_final_response()

    async def until_done(self) -> "AsyncResponseStreamWrapper[TextFormatT]":
        async for _ in self:
            pass
        return self

    def parse(self) -> "AsyncResponseStreamWrapper[TextFormatT]":
        """Called when using with_raw_response with stream=True."""
        return self

    @property
    def stream(self) -> "AsyncResponseStream[TextFormatT]":
        return self._self_stream

    @stream.setter
    def stream(self, stream: "AsyncResponseStream[TextFormatT]") -> None:
        self.__wrapped__ = stream
        self._self_stream = stream
        self._self_aiter = aiter(stream)

    @property
    def response(self):
        response = _get_stream_response(self.stream)
        if response is None:
            return None
        return _AsyncResponseProxy(response, lambda: self._stop(None))


class AsyncResponseStreamManagerWrapper(Generic[TextFormatT]):
    """Wrapper for async OpenAI Responses API stream managers."""

    def __init__(
        self,
        manager: "AsyncResponseStreamManager[TextFormatT]",
        invocation,
        capture_content: bool,
    ):
        self._manager = manager
        self._invocation = invocation
        self._capture_content = capture_content
        self._stream_wrapper: (
            AsyncResponseStreamWrapper[TextFormatT] | None
        ) = None

    async def __aenter__(self) -> AsyncResponseStreamWrapper[TextFormatT]:
        stream = await self._manager.__aenter__()
        self._stream_wrapper = AsyncResponseStreamWrapper(
            stream,
            self._invocation,
            self._capture_content,
        )
        return self._stream_wrapper

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        suppressed = False
        stream_wrapper = self._stream_wrapper
        self._stream_wrapper = None
        async with AsyncExitStack() as cleanup:
            if stream_wrapper is not None:

                async def finalize_stream_wrapper() -> None:
                    if suppressed:
                        await stream_wrapper.__aexit__(None, None, None)
                    else:
                        await stream_wrapper.__aexit__(
                            exc_type, exc_val, exc_tb
                        )

                cleanup.push_async_callback(finalize_stream_wrapper)
            suppressed = await self._manager.__aexit__(
                exc_type, exc_val, exc_tb
            )
            return suppressed

    def parse(self) -> "AsyncResponseStreamManagerWrapper[TextFormatT]":
        raise NotImplementedError(
            "AsyncResponseStreamManagerWrapper.parse() is not implemented"
        )

    # TODO: Replace __getattr__ passthrough with wrapt.ObjectProxy in a future
    # cleanup once wrapt 2 typing support is available (wrapt PR #3903).
    def __getattr__(self, name: str):
        return getattr(self._manager, name)
