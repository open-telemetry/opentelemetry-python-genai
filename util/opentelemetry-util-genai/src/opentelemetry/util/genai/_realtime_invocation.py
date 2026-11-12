# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from opentelemetry._logs import Logger, LogRecord, SeverityNumber
from opentelemetry.context import Context, get_current
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import Error, GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    ErrorTypeResolver,
)
from opentelemetry.util.types import AttributeValue

REALTIME_INFERENCE_OPERATION_NAME: Final = "realtime_inference"
USER_SPEECH_OPERATION_NAME: Final = "user_speech"
REALTIME_SESSION_STARTED_EVENT_NAME: Final = (
    "gen_ai.client.realtime_session.started"
)
REALTIME_SESSION_ENDED_EVENT_NAME: Final = (
    "gen_ai.client.realtime_session.ended"
)
GEN_AI_REALTIME_SESSION_ID: Final = "gen_ai.realtime_session.id"


@dataclass
class UserSpeechData:
    """Typed data container for a user_speech invocation."""

    provider_name: str | None = None
    realtime_session_id: str | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class UserSpeechInvocation(GenAIInvocation):
    """Represents a user_speech span capturing a user speech utterance."""

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        provider: str | None = None,
        realtime_session_id: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        start_attributes: dict[str, AttributeValue] = {}
        if provider is not None:
            start_attributes[GenAI.GEN_AI_PROVIDER_NAME] = provider
        self._data: UserSpeechData = UserSpeechData(
            provider_name=provider,
            realtime_session_id=realtime_session_id,
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=USER_SPEECH_OPERATION_NAME,
            span_name=USER_SPEECH_OPERATION_NAME,
            span_kind=SpanKind.INTERNAL,
            error_type_resolver=error_type_resolver,
            start_attributes=start_attributes,
            context=context,
            _attach_to_context=_attach_to_context,
            content_capturing_mode=content_capturing_mode,
            attributes=self._data.attributes,
            metric_attributes=self._data.metric_attributes,
        )
        self._data.attributes = self.attributes
        self._data.metric_attributes = self.metric_attributes

    @property
    def provider(self) -> str | None:
        return self._data.provider_name

    @property
    def realtime_session_id(self) -> str | None:
        return self._data.realtime_session_id

    @realtime_session_id.setter
    def realtime_session_id(self, value: str | None) -> None:
        self._data.realtime_session_id = value

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        self._data.attributes = self.attributes
        self._data.metric_attributes = self.metric_attributes
        attributes: dict[str, AttributeValue] = {}
        if self._data.realtime_session_id is not None:
            attributes[GEN_AI_REALTIME_SESSION_ID] = (
                self._data.realtime_session_id
            )
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)


def build_realtime_session_attributes(
    provider: str,
    *,
    request_model: str | None = None,
    realtime_session_id: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    error_type: str | None = None,
) -> dict[str, AttributeValue]:
    """Build attributes for realtime session started/ended events."""
    optional_attrs = (
        (GenAI.GEN_AI_PROVIDER_NAME, provider),
        (GenAI.GEN_AI_REQUEST_MODEL, request_model),
        (GEN_AI_REALTIME_SESSION_ID, realtime_session_id),
        (server_attributes.SERVER_ADDRESS, server_address),
        (server_attributes.SERVER_PORT, server_port),
        (error_attributes.ERROR_TYPE, error_type),
    )
    return {k: v for k, v in optional_attrs if v is not None}


def emit_realtime_session_started_event(
    logger: Logger,
    provider: str,
    *,
    request_model: str | None = None,
    realtime_session_id: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    context: Context | None = None,
) -> None:
    """Emit a ``gen_ai.client.realtime_session.started`` event."""
    ctx = context if context is not None else get_current()
    if not logger.enabled(
        context=ctx,
        severity_number=SeverityNumber.INFO,
        event_name=REALTIME_SESSION_STARTED_EVENT_NAME,
    ):
        return
    attributes = build_realtime_session_attributes(
        provider,
        request_model=request_model,
        realtime_session_id=realtime_session_id,
        server_address=server_address,
        server_port=server_port,
    )
    logger.emit(
        LogRecord(
            event_name=REALTIME_SESSION_STARTED_EVENT_NAME,
            severity_number=SeverityNumber.INFO,
            attributes=attributes,
            context=ctx,
        )
    )


def emit_realtime_session_ended_event(
    logger: Logger,
    provider: str,
    *,
    request_model: str | None = None,
    realtime_session_id: str | None = None,
    server_address: str | None = None,
    server_port: int | None = None,
    error: Error | BaseException | None = None,
    error_type_resolver: ErrorTypeResolver | None = None,
    context: Context | None = None,
) -> None:
    """Emit a ``gen_ai.client.realtime_session.ended`` event."""
    ctx = context if context is not None else get_current()
    severity = (
        SeverityNumber.WARN if error is not None else SeverityNumber.INFO
    )
    if not logger.enabled(
        context=ctx,
        severity_number=severity,
        event_name=REALTIME_SESSION_ENDED_EVENT_NAME,
    ):
        return
    resolved_error_type: str | None = None
    if error is not None:
        if isinstance(error, BaseException):
            error = Error.from_exception(error, error_type_resolver)
        resolved_error_type = error.type
    attributes = build_realtime_session_attributes(
        provider,
        request_model=request_model,
        realtime_session_id=realtime_session_id,
        server_address=server_address,
        server_port=server_port,
        error_type=resolved_error_type,
    )
    logger.emit(
        LogRecord(
            event_name=REALTIME_SESSION_ENDED_EVENT_NAME,
            severity_number=severity,
            attributes=attributes,
            context=ctx,
        )
    )
