# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI hooks for the shared ``with_raw_response`` proxy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from openai import AsyncStream, Stream
from openai._models import construct_type
from openai.types.chat import ChatCompletion

from opentelemetry.util.genai.raw_response import (
    RawResponseProxy,
    is_raw_response,
)
from opentelemetry.util.genai.raw_response import (
    wrap_raw_response as _wrap_raw_response,
)

from .response_extractors import (
    Response,
    get_response_error,
    set_fetch_response_attributes,
    set_invocation_response_attributes,
)
from .utils import get_served_model

if TYPE_CHECKING:
    from opentelemetry.util.genai.invocation import Error, GenAIInvocation

_logger = logging.getLogger(__name__)

#: Builds the instrumented wrapper for a parsed stream.
StreamWrapperFactory = Callable[[Any, "GenAIInvocation", bool], object]


class _OpenAIRawResponseProxy(RawResponseProxy):
    """Shared plumbing for the operations that accept ``with_raw_response``.

    Subclasses name the payload they expect and record it; everything else --
    the served-model header, the stream wrapping, and constructing the payload
    out of an already-read body -- is the same for all of them.
    """

    #: The SDK model the operation's payload parses into. ``None`` on SDK
    #: versions that do not ship the type at all, which disables extraction.
    payload_type: ClassVar[type[Any] | None] = None
    #: The body's own ``object`` discriminator for that payload.
    payload_object: ClassVar[str] = ""

    def __init__(
        self,
        raw_response: object,
        invocation: GenAIInvocation,
        capture_content: bool,
        stream_wrapper_cls: StreamWrapperFactory,
    ) -> None:
        super().__init__(raw_response, invocation)
        self._self_capture_content = capture_content
        self._self_stream_wrapper_cls = stream_wrapper_cls
        # The served model is on the headers, which are available now; a caller
        # that never parses would otherwise finalize without a response model.
        self._apply_served_model()

    def _record(self, payload: object) -> None:
        """Record the payload. Overridden where a payload can report failure."""
        self._extract(payload)
        self._apply_served_model()

    def _extract(self, payload: object) -> None:
        """Apply the operation's response attributes to the invocation."""
        raise NotImplementedError

    def _apply_served_model(self) -> None:
        """Prefer the gateway's served model over the one in the body.

        The served-model header is only on the HTTP response, which the parsed
        payload does not carry, so it is applied from the raw response after
        the payload has been recorded.
        """
        served_model = get_served_model(
            getattr(self.__wrapped__, "headers", None)
        )
        if served_model:
            self._self_invocation.response_model_name = served_model

    def _extract_from_body(self, body: dict[str, object]) -> bool:
        payload = self._payload_from_body(body)
        if payload is None:
            return False
        self._record(payload)
        return True

    def _extract_from_parsed(self, parsed: object) -> bool:
        if self.payload_type is None or not isinstance(
            parsed, self.payload_type
        ):
            return False
        self._record(parsed)
        return True

    def _wrap_parsed_stream(self, parsed: object) -> object | None:
        if not isinstance(parsed, (Stream, AsyncStream)):
            return None
        self._apply_served_model()
        return self._self_stream_wrapper_cls(
            parsed, self._self_invocation, self._self_capture_content
        )

    def _body_is_payload(self, body: dict[str, object]) -> bool:
        """Whether ``body`` is this operation's payload rather than an error."""
        discriminator = body.get("object")
        if discriminator is not None:
            return discriminator == self.payload_object
        return isinstance(body.get("id"), str)

    def _payload_from_body(self, body: dict[str, object]) -> object | None:
        """Construct this operation's payload from an already-decoded body.

        Used instead of ``parse()`` so telemetry never runs the caller's
        deferred parse, which would apply their cast target and post-parser and
        populate the SDK's parse cache. Goes through the SDK's own
        ``construct_type``, the same non-strict path ``parse()`` takes, so a
        field the installed SDK version does not know yet does not cost the
        whole response's telemetry.

        ``construct_type`` does not validate, so the body is screened first:
        without that, an error envelope would construct into a payload with
        every field ``None`` and be recorded as a successful response.

        The screen is deliberately not a bare ``object`` check. OpenAI-
        compatible gateways routinely omit the discriminator, and requiring it
        would drop the whole response's telemetry for a body the SDK itself
        parses without complaint. A body carrying no discriminator is accepted
        on the strength of its ``id``, which every payload has and an error
        envelope does not.
        """
        if self.payload_type is None:
            return None
        if not self._body_is_payload(body):
            _logger.debug(
                "raw-response body is not a %s; skipping response telemetry "
                "for this call",
                self.payload_object or "known payload",
            )
            return None
        try:
            return construct_type(type_=self.payload_type, value=body)
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.debug(
                "could not construct the raw-response payload; skipping "
                "response telemetry for this call",
                exc_info=True,
            )
            return None


class ChatRawResponseProxy(_OpenAIRawResponseProxy):
    """A raw ``chat.completions.create`` response."""

    payload_type: ClassVar[type[Any] | None] = ChatCompletion
    payload_object: ClassVar[str] = "chat.completion"

    def __init__(
        self,
        raw_response: object,
        invocation: GenAIInvocation,
        capture_content: bool,
        stream_wrapper_cls: StreamWrapperFactory,
        extract: Callable[[Any, Any, bool], object],
    ) -> None:
        # ``_set_response_properties`` lives in patch.py, which imports this
        # module, so it is injected rather than imported.
        self._self_extract = extract
        super().__init__(
            raw_response, invocation, capture_content, stream_wrapper_cls
        )

    def _extract(self, payload: object) -> None:
        self._self_extract(
            self._self_invocation, payload, self._self_capture_content
        )


class ResponsesRawResponseProxy(_OpenAIRawResponseProxy):
    """A raw ``responses.create`` response, which can report its own failure."""

    payload_type: ClassVar[type[Any] | None] = Response
    payload_object: ClassVar[str] = "response"

    def __init__(
        self,
        raw_response: object,
        invocation: GenAIInvocation,
        capture_content: bool,
        stream_wrapper_cls: StreamWrapperFactory,
    ) -> None:
        # A Responses payload can report a failed generation, which ends the
        # invocation with fail() rather than stop().
        self._self_error: Error | None = None
        super().__init__(
            raw_response, invocation, capture_content, stream_wrapper_cls
        )

    def _extract(self, payload: object) -> None:
        set_invocation_response_attributes(
            self._self_invocation, payload, self._self_capture_content
        )

    def _record(self, payload: object) -> None:
        super()._record(payload)
        self._self_error = get_response_error(payload)

    def _finalize_once(self) -> None:
        # Same finalize-at-most-once contract as the base, but a payload that
        # reported a failed generation ends the invocation as an error.
        if not self._self_span_open:
            return
        if self._self_error is None:
            super()._finalize_once()
            return
        self._self_span_open = False
        self._self_invocation.fail(self._self_error)


class FetchResponseRawResponseProxy(_OpenAIRawResponseProxy):
    """A raw ``responses.retrieve`` response.

    A replayed failure describes the original generation, not this fetch, so
    unlike ``ResponsesRawResponseProxy`` it never fails the invocation.
    """

    payload_type: ClassVar[type[Any] | None] = Response
    payload_object: ClassVar[str] = "response"

    def _extract(self, payload: object) -> None:
        set_fetch_response_attributes(
            self._self_invocation, payload, self._self_capture_content
        )


def wrap_chat_raw_response(
    result: object,
    invocation: GenAIInvocation,
    capture_content: bool,
    *,
    stream_wrapper_cls: StreamWrapperFactory,
    extract: Callable[[Any, Any, bool], object],
    streamed: bool = False,
) -> Any:
    """Wrap a raw chat-completions result, deferring ``parse()``."""
    return _wrap_raw_response(
        lambda raw: ChatRawResponseProxy(
            raw, invocation, capture_content, stream_wrapper_cls, extract
        ),
        result,
        streamed=streamed,
    )


def wrap_responses_raw_response(
    result: object,
    invocation: GenAIInvocation,
    capture_content: bool,
    *,
    stream_wrapper_cls: StreamWrapperFactory,
    fetch: bool = False,
    streamed: bool = False,
) -> Any:
    """Wrap a raw Responses API result, deferring ``parse()``."""
    proxy_cls = (
        FetchResponseRawResponseProxy if fetch else ResponsesRawResponseProxy
    )
    return _wrap_raw_response(
        lambda raw: proxy_cls(
            raw, invocation, capture_content, stream_wrapper_cls
        ),
        result,
        streamed=streamed,
    )


__all__ = [
    "ChatRawResponseProxy",
    "FetchResponseRawResponseProxy",
    "ResponsesRawResponseProxy",
    "is_raw_response",
    "wrap_chat_raw_response",
    "wrap_responses_raw_response",
]
