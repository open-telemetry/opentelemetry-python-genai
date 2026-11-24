# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections.abc import Sequence
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
from opentelemetry.util.genai._invocation import (
    _CACHE_READ_MODALITY_FIELDS,
    _GEN_AI_REQUEST_REASONING_LEVEL,
    _INPUT_MODALITY_FIELDS,
    _OUTPUT_MODALITY_FIELDS,
    Error,
    GenAIInvocation,
    _get_token_usage_attributes,
    _is_active_completion_hook,
    _resolve_finish_reasons,
    _set_modality_tokens,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    ModalityTokens,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

# TODO: Migrate to gen_ai_attributes constants once available in the semconv
# package. Added to the GenAI semantic conventions in
# https://github.com/open-telemetry/semantic-conventions-genai/pull/394.
_REALTIME_INFERENCE_OPERATION_NAME: Final = "realtime_inference"
_USER_SPEECH_OPERATION_NAME: Final = "user_speech"
_REALTIME_SESSION_STARTED_EVENT_NAME: Final = (
    "gen_ai.client.realtime_session.started"
)
_REALTIME_SESSION_ENDED_EVENT_NAME: Final = (
    "gen_ai.client.realtime_session.ended"
)
_GEN_AI_REALTIME_SESSION_ID: Final = "gen_ai.realtime_session.id"


def should_capture_realtime_content(
    content_capturing_mode: ContentCapturingMode,
    completion_hook: CompletionHook,
) -> bool:
    """Return True when realtime_inference content should be captured.

    Mirrors the capture policy of :class:`RealtimeInferenceInvocation` before
    a generation turn starts so instrumentations can decide whether to buffer
    incoming media or messages without reading private handler internals.
    """
    return content_capturing_mode in (
        ContentCapturingMode.SPAN_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    ) or _is_active_completion_hook(completion_hook)


@dataclass
class _RealtimeInferenceData:
    """Typed data container for realtime inference invocations."""

    provider_name: str | None = None
    request_model: str | None = None
    server_address: str | None = None
    server_port: int | None = None
    realtime_session_id: str | None = None
    output_type: str | None = None
    request_max_tokens: int | None = None
    request_reasoning_level: str | None = None
    response_finish_reasons: list[str] | None = None
    response_id: str | None = None
    response_model: str | None = None
    input_messages: Sequence[InputMessage] | None = None
    output_messages: Sequence[OutputMessage] | None = None
    system_instructions: (
        Sequence[SystemInstructionPart] | Sequence[MessagePart] | None
    ) = None
    tool_definitions: list[ToolDefinition] | None = None
    usage_audio_cache_read_input_tokens: int | None = None
    usage_audio_input_tokens: int | None = None
    usage_audio_output_tokens: int | None = None
    usage_cache_read_input_tokens: int | None = None
    usage_cache_write_input_tokens: int | None = None
    usage_image_cache_read_input_tokens: int | None = None
    usage_image_input_tokens: int | None = None
    usage_image_output_tokens: int | None = None
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_reasoning_output_tokens: int | None = None
    usage_text_cache_read_input_tokens: int | None = None
    usage_text_input_tokens: int | None = None
    usage_text_output_tokens: int | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class RealtimeInferenceInvocation(GenAIInvocation):
    """Represents a single realtime inference generation turn.

    Use ``handler.realtime_inference(provider)`` rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-realtime.md#realtime-content-generation

    Emits a ``CLIENT`` span named ``realtime_inference {gen_ai.request.model}``
    (or ``realtime_inference`` when ``request_model`` is unset) carrying the
    attributes defined by the realtime semantic conventions, including
    ``gen_ai.realtime_session.id`` (set at span start when provided, or updated
    on finish if assigned afterward), token usage, and opt-in message content.
    Unlike ``InferenceInvocation``, ``realtime_inference`` emits no client
    metrics, no ``gen_ai.client.inference.operation.details`` event, and no
    ``gen_ai.client.inference.exception`` event.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        realtime_session_id: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
                (_GEN_AI_REALTIME_SESSION_ID, realtime_session_id),
            )
            if v is not None
        }
        self._data: _RealtimeInferenceData = _RealtimeInferenceData(
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            realtime_session_id=realtime_session_id,
            input_messages=[],
            output_messages=[],
            system_instructions=[],
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_REALTIME_INFERENCE_OPERATION_NAME,
            span_name=f"{_REALTIME_INFERENCE_OPERATION_NAME} {request_model}"
            if request_model
            else _REALTIME_INFERENCE_OPERATION_NAME,
            span_kind=SpanKind.CLIENT,
            error_type_resolver=error_type_resolver,
            start_attributes=start_attributes,
            context=context,
            conversation_id=conversation_id,
            content_capturing_mode=content_capturing_mode,
            _attach_to_context=_attach_to_context,
            attributes=self._data.attributes,
        )
        self._data.attributes = self.attributes
        self._end_time_ns: int | None = None

    def mark_end(self) -> None:
        """Record the operation end timestamp before finalization.

        Used when a completion or interruption signal arrives before final
        metadata (such as trailing usage), so the span end timestamp reflects
        the observed operation end rather than finalization time.
        """
        if self._finished or self._end_time_ns is not None:
            return
        self._end_time_ns = time.time_ns()

    @property
    def provider(self) -> str | None:
        return self._data.provider_name

    @property
    def request_model(self) -> str | None:
        return self._data.request_model

    @property
    def server_address(self) -> str | None:
        return self._data.server_address

    @property
    def server_port(self) -> int | None:
        return self._data.server_port

    @property
    def realtime_session_id(self) -> str | None:
        return self._data.realtime_session_id

    @realtime_session_id.setter
    def realtime_session_id(self, value: str | None) -> None:
        self._data.realtime_session_id = value

    @property
    def response_model_name(self) -> str | None:
        return self._data.response_model

    @response_model_name.setter
    def response_model_name(self, value: str | None) -> None:
        self._data.response_model = value

    @property
    def response_id(self) -> str | None:
        return self._data.response_id

    @response_id.setter
    def response_id(self, value: str | None) -> None:
        self._data.response_id = value

    @property
    def finish_reasons(self) -> list[str] | None:
        return self._data.response_finish_reasons

    @finish_reasons.setter
    def finish_reasons(self, value: Sequence[str] | None) -> None:
        self._data.response_finish_reasons = (
            list(value) if value is not None else None
        )

    @property
    def output_type(self) -> str | None:
        return self._data.output_type

    @output_type.setter
    def output_type(self, value: str | None) -> None:
        self._data.output_type = value

    @property
    def max_tokens(self) -> int | None:
        return self._data.request_max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None) -> None:
        self._data.request_max_tokens = value

    @property
    def reasoning_level(self) -> str | None:
        return self._data.request_reasoning_level

    @reasoning_level.setter
    def reasoning_level(self, value: str | None) -> None:
        self._data.request_reasoning_level = value

    @property
    def input_tokens(self) -> int | None:
        return self._data.usage_input_tokens

    @input_tokens.setter
    def input_tokens(self, value: int | None) -> None:
        self._data.usage_input_tokens = value

    @property
    def output_tokens(self) -> int | None:
        return self._data.usage_output_tokens

    @output_tokens.setter
    def output_tokens(self, value: int | None) -> None:
        self._data.usage_output_tokens = value

    @property
    def thinking_tokens(self) -> int | None:
        return self._data.usage_reasoning_output_tokens

    @thinking_tokens.setter
    def thinking_tokens(self, value: int | None) -> None:
        self._data.usage_reasoning_output_tokens = value

    @property
    def cache_write_input_tokens(self) -> int | None:
        return self._data.usage_cache_write_input_tokens

    @cache_write_input_tokens.setter
    def cache_write_input_tokens(self, value: int | None) -> None:
        self._data.usage_cache_write_input_tokens = value

    @property
    def cache_read_input_tokens(self) -> int | None:
        return self._data.usage_cache_read_input_tokens

    @cache_read_input_tokens.setter
    def cache_read_input_tokens(self, value: int | None) -> None:
        self._data.usage_cache_read_input_tokens = value

    @property
    def text_input_tokens(self) -> int | None:
        return self._data.usage_text_input_tokens

    @text_input_tokens.setter
    def text_input_tokens(self, value: int | None) -> None:
        self._data.usage_text_input_tokens = value

    @property
    def image_input_tokens(self) -> int | None:
        return self._data.usage_image_input_tokens

    @image_input_tokens.setter
    def image_input_tokens(self, value: int | None) -> None:
        self._data.usage_image_input_tokens = value

    @property
    def audio_input_tokens(self) -> int | None:
        return self._data.usage_audio_input_tokens

    @audio_input_tokens.setter
    def audio_input_tokens(self, value: int | None) -> None:
        self._data.usage_audio_input_tokens = value

    @property
    def text_output_tokens(self) -> int | None:
        return self._data.usage_text_output_tokens

    @text_output_tokens.setter
    def text_output_tokens(self, value: int | None) -> None:
        self._data.usage_text_output_tokens = value

    @property
    def image_output_tokens(self) -> int | None:
        return self._data.usage_image_output_tokens

    @image_output_tokens.setter
    def image_output_tokens(self, value: int | None) -> None:
        self._data.usage_image_output_tokens = value

    @property
    def audio_output_tokens(self) -> int | None:
        return self._data.usage_audio_output_tokens

    @audio_output_tokens.setter
    def audio_output_tokens(self, value: int | None) -> None:
        self._data.usage_audio_output_tokens = value

    @property
    def text_cache_read_input_tokens(self) -> int | None:
        return self._data.usage_text_cache_read_input_tokens

    @text_cache_read_input_tokens.setter
    def text_cache_read_input_tokens(self, value: int | None) -> None:
        self._data.usage_text_cache_read_input_tokens = value

    @property
    def image_cache_read_input_tokens(self) -> int | None:
        return self._data.usage_image_cache_read_input_tokens

    @image_cache_read_input_tokens.setter
    def image_cache_read_input_tokens(self, value: int | None) -> None:
        self._data.usage_image_cache_read_input_tokens = value

    @property
    def audio_cache_read_input_tokens(self) -> int | None:
        return self._data.usage_audio_cache_read_input_tokens

    @audio_cache_read_input_tokens.setter
    def audio_cache_read_input_tokens(self, value: int | None) -> None:
        self._data.usage_audio_cache_read_input_tokens = value

    @property
    def input_messages(self) -> list[InputMessage]:
        if isinstance(self._data.input_messages, list):
            return self._data.input_messages
        messages = (
            list(self._data.input_messages)
            if self._data.input_messages
            else []
        )
        self._data.input_messages = messages
        return messages

    @input_messages.setter
    def input_messages(self, value: Sequence[InputMessage]) -> None:
        self._data.input_messages = value

    @property
    def output_messages(self) -> list[OutputMessage]:
        if isinstance(self._data.output_messages, list):
            return self._data.output_messages
        messages = (
            list(self._data.output_messages)
            if self._data.output_messages
            else []
        )
        self._data.output_messages = messages
        return messages

    @output_messages.setter
    def output_messages(self, value: Sequence[OutputMessage]) -> None:
        self._data.output_messages = value

    @property
    def system_instruction(
        self,
    ) -> list[SystemInstructionPart] | list[MessagePart]:
        """System instructions for the model. Passing ``MessagePart`` is deprecated; use ``SystemInstructionPart``."""
        if isinstance(self._data.system_instructions, list):
            return self._data.system_instructions
        instructions = (
            list(self._data.system_instructions)
            if self._data.system_instructions
            else []
        )
        self._data.system_instructions = instructions
        return instructions

    @system_instruction.setter
    def system_instruction(
        self,
        value: Sequence[SystemInstructionPart] | Sequence[MessagePart] | None,
    ) -> None:
        self._data.system_instructions = (
            list(value) if value is not None else None
        )

    @property
    def tool_definitions(self) -> list[ToolDefinition] | None:
        return self._data.tool_definitions

    @tool_definitions.setter
    def tool_definitions(self, value: Sequence[ToolDefinition] | None) -> None:
        self._data.tool_definitions = (
            list(value) if value is not None else None
        )

    def set_input_tokens(self, entries: ModalityTokens | None) -> None:
        """Record the per-modality breakdown of the input tokens."""
        _set_modality_tokens(self._data, _INPUT_MODALITY_FIELDS, entries)

    def set_output_tokens(self, entries: ModalityTokens | None) -> None:
        """Record the per-modality breakdown of the output tokens."""
        _set_modality_tokens(self._data, _OUTPUT_MODALITY_FIELDS, entries)

    def set_cache_read_input_tokens(
        self, entries: ModalityTokens | None
    ) -> None:
        """Record the per-modality breakdown of the cache read input tokens."""
        _set_modality_tokens(self._data, _CACHE_READ_MODALITY_FIELDS, entries)

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for realtime_inference.

        Because ``realtime_inference`` does not emit a
        ``gen_ai.client.inference.operation.details`` event, ``EVENT_ONLY`` mode
        captures content only when a completion hook is active.
        """
        return self._should_capture_content_on_span or (
            _is_active_completion_hook(self._completion_hook)
        )

    def _get_attributes(self) -> dict[str, AttributeValue]:
        finish_reasons = _resolve_finish_reasons(
            self._data.response_finish_reasons,
            self._data.output_messages,
        )
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_PROVIDER_NAME, self._data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, self._data.request_model),
            (server_attributes.SERVER_ADDRESS, self._data.server_address),
            (server_attributes.SERVER_PORT, self._data.server_port),
            (GenAI.GEN_AI_CONVERSATION_ID, self.conversation_id),
            (GenAI.GEN_AI_OUTPUT_TYPE, self._data.output_type),
            (_GEN_AI_REALTIME_SESSION_ID, self._data.realtime_session_id),
            (GenAI.GEN_AI_REQUEST_MAX_TOKENS, self._data.request_max_tokens),
            (
                _GEN_AI_REQUEST_REASONING_LEVEL,
                self._data.request_reasoning_level,
            ),
            (GenAI.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons),
            (GenAI.GEN_AI_RESPONSE_ID, self._data.response_id),
            (GenAI.GEN_AI_RESPONSE_MODEL, self._data.response_model),
        )
        attrs: dict[str, AttributeValue] = {
            k: v for k, v in optional_attrs if v is not None
        }
        attrs.update(
            _get_token_usage_attributes(
                input_tokens=self._data.usage_input_tokens,
                output_tokens=self._data.usage_output_tokens,
                cache_write_input_tokens=self._data.usage_cache_write_input_tokens,
                cache_read_input_tokens=self._data.usage_cache_read_input_tokens,
                reasoning_output_tokens=self._data.usage_reasoning_output_tokens,
                text_input_tokens=self._data.usage_text_input_tokens,
                image_input_tokens=self._data.usage_image_input_tokens,
                audio_input_tokens=self._data.usage_audio_input_tokens,
                text_output_tokens=self._data.usage_text_output_tokens,
                image_output_tokens=self._data.usage_image_output_tokens,
                audio_output_tokens=self._data.usage_audio_output_tokens,
                text_cache_read_input_tokens=self._data.usage_text_cache_read_input_tokens,
                image_cache_read_input_tokens=self._data.usage_image_cache_read_input_tokens,
                audio_cache_read_input_tokens=self._data.usage_audio_cache_read_input_tokens,
            )
        )
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        self._data.attributes = self.attributes
        if self.span.is_recording():
            attributes: dict[str, AttributeValue] = {}
            attributes.update(self._get_attributes())
            attributes.update(
                get_content_attributes(
                    input_messages=self.input_messages,
                    output_messages=self.output_messages,
                    system_instruction=self.system_instruction,
                    tool_definitions=self.tool_definitions,
                    for_span=True,
                    content_capturing_mode=self._content_capturing_mode,
                )
            )
            attributes.update(self.attributes)
            span_attributes = {
                k: v
                for k, v in attributes.items()
                if k not in self._start_attributes
                or self._start_attributes[k] != v
            }
            if span_attributes:
                self.span.set_attributes(span_attributes)
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
            log_record=None,
        )

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        try:
            self._apply_finish(error)
        finally:
            self._detach()
            self.span.end(end_time=self._end_time_ns)


@dataclass
class _UserSpeechData:
    """Typed data container for a user_speech invocation."""

    provider_name: str | None = None
    realtime_session_id: str | None = None


class UserSpeechInvocation(GenAIInvocation):
    """Represents a user_speech span capturing a user speech utterance.

    Use ``handler.user_speech()`` rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-realtime.md#user-speech-span

    Semantic convention attributes for user speech spans:

    - ``gen_ai.operation.name``: ``"user_speech"`` (Required)
    - ``gen_ai.provider.name``: Set at span start only when ``provider`` is provided (Recommended)
    - ``gen_ai.realtime_session.id``: Set at span start when ``realtime_session_id`` is provided,
      or updated on finish if assigned afterward (Conditionally Required)
    - ``error.type``: Set when the operation ends in an error (Conditionally Required)
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        *,
        provider: str | None = None,
        realtime_session_id: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (_GEN_AI_REALTIME_SESSION_ID, realtime_session_id),
            )
            if v is not None
        }
        self._data: _UserSpeechData = _UserSpeechData(
            provider_name=provider,
            realtime_session_id=realtime_session_id,
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            _NoOpCompletionHook(),
            operation_name=_USER_SPEECH_OPERATION_NAME,
            span_name=_USER_SPEECH_OPERATION_NAME,
            span_kind=SpanKind.INTERNAL,
            error_type_resolver=error_type_resolver,
            start_attributes=start_attributes,
            context=context,
            _attach_to_context=_attach_to_context,
            content_capturing_mode=content_capturing_mode,
        )

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
        if not self.span.is_recording():
            return
        attributes: dict[str, AttributeValue] = {}
        if (
            self._data.realtime_session_id is not None
            and self._data.realtime_session_id
            != self._start_attributes.get(_GEN_AI_REALTIME_SESSION_ID)
        ):
            attributes[_GEN_AI_REALTIME_SESSION_ID] = (
                self._data.realtime_session_id
            )
        attributes.update(self.attributes)
        if attributes:
            self.span.set_attributes(attributes)


def _build_realtime_session_attributes(
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
        (_GEN_AI_REALTIME_SESSION_ID, realtime_session_id),
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
    """Emit a ``gen_ai.client.realtime_session.started`` event (INFO severity).

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-realtime.md#event-gen_aiclientrealtime_sessionstarted

    Sets ``gen_ai.provider.name`` (Required) and, when provided,
    ``gen_ai.request.model``, ``gen_ai.realtime_session.id``,
    ``server.address``, and ``server.port``.
    """
    ctx = context if context is not None else get_current()
    if not logger.enabled(
        context=ctx,
        severity_number=SeverityNumber.INFO,
        event_name=_REALTIME_SESSION_STARTED_EVENT_NAME,
    ):
        return
    attributes = _build_realtime_session_attributes(
        provider,
        request_model=request_model,
        realtime_session_id=realtime_session_id,
        server_address=server_address,
        server_port=server_port,
    )
    logger.emit(
        LogRecord(
            event_name=_REALTIME_SESSION_STARTED_EVENT_NAME,
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
    """Emit a ``gen_ai.client.realtime_session.ended`` event.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-realtime.md#event-gen_aiclientrealtime_sessionended

    Sets severity to ``INFO`` on normal close and ``WARN`` when ``error`` is
    provided. Sets ``gen_ai.provider.name`` (Required) and, when provided,
    ``gen_ai.request.model``, ``gen_ai.realtime_session.id``,
    ``server.address``, ``server.port``, and ``error.type`` (resolved from
    ``error``).
    """
    ctx = context if context is not None else get_current()
    severity = (
        SeverityNumber.WARN if error is not None else SeverityNumber.INFO
    )
    if not logger.enabled(
        context=ctx,
        severity_number=severity,
        event_name=_REALTIME_SESSION_ENDED_EVENT_NAME,
    ):
        return
    resolved_error_type: str | None = None
    if error is not None:
        if isinstance(error, BaseException):
            error = Error.from_exception(error, error_type_resolver)
        resolved_error_type = error.type
    attributes = _build_realtime_session_attributes(
        provider,
        request_model=request_model,
        realtime_session_id=realtime_session_id,
        server_address=server_address,
        server_port=server_port,
        error_type=resolved_error_type,
    )
    logger.emit(
        LogRecord(
            event_name=_REALTIME_SESSION_ENDED_EVENT_NAME,
            severity_number=severity,
            attributes=attributes,
            context=ctx,
        )
    )
