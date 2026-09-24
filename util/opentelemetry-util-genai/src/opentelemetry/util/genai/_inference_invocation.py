# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Final

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    server_attributes,
)
from opentelemetry.trace import (
    INVALID_SPAN,
    Span,
    SpanKind,
    Tracer,
)
from opentelemetry.util.genai._context import (
    INFERENCE_CONTEXT_KEY,
    get_inference_context_data,
)
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.types import (
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    _should_emit_event,
)
from opentelemetry.util.types import AttributeValue

_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: Final = (
    "gen_ai.usage.cache_write.input_tokens"
)
_GEN_AI_USAGE_TEXT_INPUT_TOKENS: Final = "gen_ai.usage.text.input_tokens"
_GEN_AI_USAGE_IMAGE_INPUT_TOKENS: Final = "gen_ai.usage.image.input_tokens"
_GEN_AI_USAGE_AUDIO_INPUT_TOKENS: Final = "gen_ai.usage.audio.input_tokens"
_GEN_AI_USAGE_TEXT_OUTPUT_TOKENS: Final = "gen_ai.usage.text.output_tokens"
_GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS: Final = "gen_ai.usage.image.output_tokens"
_GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS: Final = "gen_ai.usage.audio.output_tokens"
_GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS: Final = (
    "gen_ai.usage.text.cache_read.input_tokens"
)
_GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS: Final = (
    "gen_ai.usage.image.cache_read.input_tokens"
)
_GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS: Final = (
    "gen_ai.usage.audio.cache_read.input_tokens"
)
_INPUT_MODALITY_FIELDS: Final[Mapping[str, str]] = {
    Modality.TEXT: "text_input_tokens",
    Modality.IMAGE: "image_input_tokens",
    Modality.AUDIO: "audio_input_tokens",
}
_OUTPUT_MODALITY_FIELDS: Final[Mapping[str, str]] = {
    Modality.TEXT: "text_output_tokens",
    Modality.IMAGE: "image_output_tokens",
    Modality.AUDIO: "audio_output_tokens",
}
_CACHE_READ_MODALITY_FIELDS: Final[Mapping[str, str]] = {
    Modality.TEXT: "text_cache_read_input_tokens",
    Modality.IMAGE: "image_cache_read_input_tokens",
    Modality.AUDIO: "audio_cache_read_input_tokens",
}
_GEN_AI_REQUEST_REASONING_LEVEL: Final = "gen_ai.request.reasoning.level"
_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: Final = (
    "gen_ai.request.previous_response.id"
)
_GEN_AI_CONVERSATION_COMPACTED: Final = "gen_ai.conversation.compacted"
_GEN_AI_PROMPT_VERSION: Final = "gen_ai.prompt.version"


@dataclass
class InferenceNonContentCaptureData:
    """Typed data passed from inner inference invocations to the outer invocation."""

    provider: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    server_address: str | None = None
    server_port: int | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    seed: int | None = None
    request_choice_count: int | None = None
    output_type: str | None = None
    request_stream: bool | None = None
    ttfc_seconds: float | None = None
    cache_write_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    text_input_tokens: int | None = None
    image_input_tokens: int | None = None
    audio_input_tokens: int | None = None
    text_output_tokens: int | None = None
    image_output_tokens: int | None = None
    audio_output_tokens: int | None = None
    text_cache_read_input_tokens: int | None = None
    image_cache_read_input_tokens: int | None = None
    audio_cache_read_input_tokens: int | None = None
    reasoning_level: str | None = None
    previous_response_id: str | None = None
    conversation_compacted: bool | None = None
    prompt_name: str | None = None
    prompt_version: str | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )

    def merge(
        self, other: InferenceNonContentCaptureData, *, overwrite: bool = True
    ) -> None:
        """Merge another context data instance into this one.

        Args:
            other: The context data to merge from.
            overwrite: If True, values from ``other`` overwrite existing values
                (used when inner invocations publish to context). If False,
                existing non-None values in ``self`` are preserved (used when
                the root invocation enriches from context).
        """
        for f in fields(self):
            if f.name in ("attributes", "metric_attributes"):
                continue
            val = getattr(other, f.name)
            if val is not None and (
                overwrite or getattr(self, f.name) is None
            ):
                setattr(self, f.name, val)

        if overwrite:
            self.attributes.update(other.attributes)
            self.metric_attributes.update(other.metric_attributes)
        else:
            for k, v in other.attributes.items():
                self.attributes.setdefault(k, v)
            for k, v in other.metric_attributes.items():
                self.metric_attributes.setdefault(k, v)


@dataclass
class InferenceContentData:
    """Holds prompt and message content for an inference invocation.

    Kept local to the invocation and never propagated via Context.
    """

    input_messages: list[InputMessage] = field(
        default_factory=list[InputMessage]
    )
    output_messages: list[OutputMessage] = field(
        default_factory=list[OutputMessage]
    )
    system_instruction: list[SystemInstructionPart] | list[MessagePart] = (
        field(default_factory=list[SystemInstructionPart])
    )
    prompt_variables: Mapping[str, object] | None = None
    tool_definitions: list[ToolDefinition] | None = None


class InferenceInvocation(GenAIInvocation):
    """Represents a single LLM chat/completion call.

    Use handler.inference(provider) rather than constructing this directly.
    """

    _context_key = INFERENCE_CONTEXT_KEY
    _context_factory = InferenceNonContentCaptureData

    data: InferenceNonContentCaptureData
    content: InferenceContentData

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
        operation_name: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        start_span: bool = True,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        operation_name = (
            operation_name or GenAI.GenAiOperationNameValues.CHAT.value
        )
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
            )
            if v is not None
        }
        self.data: InferenceNonContentCaptureData = (
            InferenceNonContentCaptureData(
                provider=provider,
                request_model=request_model,
                server_address=server_address,
                server_port=server_port,
            )
        )
        self.content: InferenceContentData = InferenceContentData()
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=operation_name,
            span_name=f"{operation_name} {request_model}"
            if request_model
            else operation_name,
            span_kind=SpanKind.CLIENT,
            error_type_resolver=error_type_resolver,
            start_attributes=start_attributes,
            context=context,
            conversation_id=conversation_id,
            content_capturing_mode=content_capturing_mode,
            start_span=start_span,
            _attach_to_context=_attach_to_context,
        )
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        self._emit_event: bool = _should_emit_event(
            self._content_capturing_mode
        )

    def set_input_tokens(self, entries: ModalityTokens | None) -> None:
        """Record the per-modality breakdown of the input tokens.

        Sets ``gen_ai.usage.{text,image,audio}.input_tokens`` from
        ``entries``, an iterable of ``(modality, token count)`` pairs.
        The modality may be a plain string or an enum member carrying one as
        its ``value``; anything outside text, image and audio is dropped, as is
        a count that is not a non-negative :class:`int`.

        The breakdown is replaced wholesale, so a modality missing from
        ``entries`` is cleared. Pass ``None`` to leave the current values
        alone, which is what a streaming chunk carrying no usage should do.
        """
        self._set_modality_tokens(_INPUT_MODALITY_FIELDS, entries)

    def set_output_tokens(self, entries: ModalityTokens | None) -> None:
        """Record the per-modality breakdown of the output tokens.

        Sets ``gen_ai.usage.{text,image,audio}.output_tokens``. See
        :meth:`set_input_tokens` for the argument contract.
        """
        self._set_modality_tokens(_OUTPUT_MODALITY_FIELDS, entries)

    def set_cache_read_input_tokens(
        self, entries: ModalityTokens | None
    ) -> None:
        """Record the per-modality breakdown of the cache read input tokens.

        Sets ``gen_ai.usage.{text,image,audio}.cache_read.input_tokens``. See
        :meth:`set_input_tokens` for the argument contract.
        """
        self._set_modality_tokens(_CACHE_READ_MODALITY_FIELDS, entries)

    def _set_modality_tokens(
        self, fields: Mapping[str, str], entries: ModalityTokens | None
    ) -> None:
        if entries is None:
            return
        for field_name in fields.values():
            setattr(self.data, field_name, None)
        for modality, token_count in entries:
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                continue
            # modality may be an enum, whose str() is "MediaModality.AUDIO"
            # rather than the bare name.
            field_name = fields.get(
                str(getattr(modality, "value", modality)).lower()
            )
            if field_name is not None:
                setattr(self.data, field_name, token_count)

    def _get_message_attributes(
        self, *, for_span: bool
    ) -> dict[str, AttributeValue]:
        return get_content_attributes(
            input_messages=self.content.input_messages,
            output_messages=self.content.output_messages,
            system_instruction=self.content.system_instruction,
            tool_definitions=self.content.tool_definitions,
            prompt_variables=self.content.prompt_variables,
            for_span=for_span,
            content_capturing_mode=self._content_capturing_mode,
        )

    def _get_finish_reasons(self) -> list[str] | None:
        if self.data.finish_reasons is not None:
            return self.data.finish_reasons or None
        if self.content.output_messages:
            reasons = [
                msg.finish_reason
                for msg in self.content.output_messages
                if msg.finish_reason
            ]
            return reasons or None
        return None

    def _on_stream_chunk(self, chunk_at: float) -> None:
        super()._on_stream_chunk(chunk_at)
        self.data.request_stream = True
        self.data.ttfc_seconds = self._ttfc_seconds

    def _get_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        optional_attrs = (
            (GenAI.GEN_AI_PROVIDER_NAME, self.data.provider),
            (GenAI.GEN_AI_REQUEST_MODEL, self.data.request_model),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
            (GenAI.GEN_AI_CONVERSATION_ID, self.conversation_id),
            (GenAI.GEN_AI_REQUEST_STREAM, self.data.request_stream),
            (GenAI.GEN_AI_REQUEST_TEMPERATURE, self.data.temperature),
            (GenAI.GEN_AI_REQUEST_TOP_P, self.data.top_p),
            (GenAI.GEN_AI_REQUEST_TOP_K, self.data.top_k),
            (
                GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                self.data.frequency_penalty,
            ),
            (
                GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY,
                self.data.presence_penalty,
            ),
            (GenAI.GEN_AI_REQUEST_MAX_TOKENS, self.data.max_tokens),
            (GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, self.data.stop_sequences),
            (GenAI.GEN_AI_REQUEST_SEED, self.data.seed),
            (GenAI.GEN_AI_RESPONSE_FINISH_REASONS, self._get_finish_reasons()),
            (GenAI.GEN_AI_RESPONSE_MODEL, self.data.response_model),
            (GenAI.GEN_AI_RESPONSE_ID, self.data.response_id),
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, self.data.input_tokens),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, self.data.output_tokens),
            (
                GenAI.GEN_AI_REQUEST_CHOICE_COUNT,
                self.data.request_choice_count,
            ),
            (GenAI.GEN_AI_OUTPUT_TYPE, self.data.output_type),
            (
                _GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
                self.data.cache_write_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                self.data.cache_read_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                self.data.thinking_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_INPUT_TOKENS,
                self.data.text_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
                self.data.image_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
                self.data.audio_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
                self.data.text_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
                self.data.image_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
                self.data.audio_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
                self.data.text_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
                self.data.image_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
                self.data.audio_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_REQUEST_REASONING_LEVEL,
                self.data.reasoning_level,
            ),
            (
                _GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
                self.data.previous_response_id,
            ),
            (
                _GEN_AI_CONVERSATION_COMPACTED,
                True if self.data.conversation_compacted else None,
            ),
            (
                GenAI.GEN_AI_PROMPT_NAME,
                self.data.prompt_name,
            ),
            (
                _GEN_AI_PROMPT_VERSION,
                self.data.prompt_version,
            ),
            (
                GenAI.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
                self.data.ttfc_seconds,
            ),
        )
        attrs.update({k: v for k, v in optional_attrs if v is not None})
        return attrs

    def enrich_from_context(
        self, data: InferenceNonContentCaptureData
    ) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values.
        """
        self.data.merge(data, overwrite=False)

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs = dict(self._start_attributes)
        if self.data.provider is not None:
            attrs[GenAI.GEN_AI_PROVIDER_NAME] = self.data.provider
        if self.data.request_model is not None:
            attrs[GenAI.GEN_AI_REQUEST_MODEL] = self.data.request_model
        if self.data.response_model is not None:
            attrs[GenAI.GEN_AI_RESPONSE_MODEL] = self.data.response_model
        if self.data.server_address is not None:
            attrs[server_attributes.SERVER_ADDRESS] = self.data.server_address
        if self.data.server_port is not None:
            attrs[server_attributes.SERVER_PORT] = self.data.server_port
        attrs.update(self.metric_attributes)
        return attrs

    def _get_metric_token_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.data.input_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.INPUT.value] = (
                self.data.input_tokens
            )
        if self.data.output_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.OUTPUT.value] = (
                self.data.output_tokens
            )
        return counts

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_inference_context_data(self._span_context)
        if ctx_data is not None:
            self.enrich_from_context(ctx_data)
        attributes: dict[str, AttributeValue] = {}
        attributes.update(self._get_attributes())
        attributes.update(self._get_message_attributes(for_span=True))
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._record_client_metrics()
        log_record = self._maybe_create_event()
        self._call_completion_hook(
            inputs=self.content.input_messages,
            outputs=self.content.output_messages,
            system_instruction=self.content.system_instruction,
            tool_definitions=self.content.tool_definitions,
            log_record=log_record,
        )
        if log_record is not None:
            self._logger.emit(log_record)

    def _maybe_create_event(self) -> LogRecord | None:
        """Emit a gen_ai.client.inference.operation.details event.

        For more details, see the semantic convention documentation:
        https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-events.md#event-eventgen_aiclientinferenceoperationdetails
        """
        if not self._emit_event:
            return None

        attributes: dict[str, AttributeValue] = {}
        attributes.update(self._start_attributes)
        attributes.update(self._get_attributes())
        attributes.update(self._get_message_attributes(for_span=False))
        attributes.update(self.attributes)
        return LogRecord(
            event_name="gen_ai.client.inference.operation.details",
            attributes=attributes,
            context=self._span_context,
        )


class SuppressedInferenceInvocation(InferenceInvocation):
    """Represents an inference invocation running inside an active inference context.

    Suppresses span creation, metrics, and events. On stop or fail, publishes its
    attributes to the active inference context.
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
        operation_name: str | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
    ) -> None:
        super().__init__(
            tracer,
            instruments,
            logger,
            _NoOpCompletionHook(),
            provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            operation_name=operation_name,
            error_type_resolver=error_type_resolver,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
            start_span=False,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
        )

    def _on_stream_chunk(self, chunk_at: float) -> None:
        last_chunk_at = (
            self._stream_last_chunk_at
            if self._stream_last_chunk_at is not None
            else self._monotonic_start_s
        )
        self._stream_last_chunk_at = chunk_at
        delta = max(chunk_at - last_chunk_at, 0.0)
        if self._ttfc_seconds is None:
            self._ttfc_seconds = delta
            self.data.ttfc_seconds = delta
        self.data.request_stream = True

    def publish_to_context(self, data: InferenceNonContentCaptureData) -> None:
        """Publish invocation attributes to the active inference context."""
        data.merge(self.data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True

        if self.data.finish_reasons is None:
            self.data.finish_reasons = self._get_finish_reasons()

        ctx_data = get_inference_context_data(self._span_context)
        if ctx_data is not None:
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass


@dataclass
class LLMInvocation:
    """Deprecated. Use InferenceInvocation instead.

    Data container for an LLM invocation. Pass to handler.llm() to start
    the span, then update fields and call handler.stop_llm() or handler.fail_llm().
    """

    request_model: str | None = None
    input_messages: list[InputMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    output_messages: list[OutputMessage] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    system_instruction: (  # pyright: ignore[reportUnknownVariableType]
        list[SystemInstructionPart] | list[MessagePart]
    ) = field(default_factory=list)
    provider: str | None = None
    response_model_name: str | None = None
    response_id: str | None = None
    finish_reasons: list[str] | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    attributes: dict[str, AttributeValue] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Additional attributes to set on spans and/or events. Not set on metrics."""
    metric_attributes: dict[str, AttributeValue] = field(default_factory=dict)  # pyright: ignore[reportUnknownVariableType]
    """Additional attributes to set on metrics. Must be low cardinality. Not set on spans or events."""
    temperature: float | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    max_tokens: int | None = None
    stop_sequences: list[str] | None = None
    seed: int | None = None
    server_address: str | None = None
    server_port: int | None = None

    _inference_invocation: InferenceInvocation | None = field(
        default=None, init=False, repr=False
    )

    def _start_with_handler(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Create and start an InferenceInvocation from this data container. Called by handler.start_llm()."""
        invocation_cls: type[InferenceInvocation] = (
            SuppressedInferenceInvocation
            if get_inference_context_data() is not None
            else InferenceInvocation
        )
        inv = invocation_cls(
            tracer,
            instruments,
            logger,
            completion_hook,
            self.provider or "",
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            content_capturing_mode=content_capturing_mode,
        )
        inv.content.input_messages = self.input_messages
        inv.content.output_messages = self.output_messages
        inv.content.system_instruction = self.system_instruction
        inv.data.response_model = self.response_model_name
        inv.data.response_id = self.response_id
        inv.data.finish_reasons = self.finish_reasons
        inv.data.input_tokens = self.input_tokens
        inv.data.output_tokens = self.output_tokens

        inv.data.temperature = self.temperature
        inv.data.top_p = self.top_p
        inv.data.frequency_penalty = self.frequency_penalty
        inv.data.presence_penalty = self.presence_penalty
        inv.data.max_tokens = self.max_tokens
        inv.data.stop_sequences = self.stop_sequences
        inv.data.seed = self.seed
        inv.attributes.update(self.attributes)
        inv.metric_attributes.update(self.metric_attributes)
        self._inference_invocation = inv

    def _sync_to_invocation(self) -> None:
        inv = self._inference_invocation
        if inv is None:
            return
        # Start attributes (provider, request_model, server_address, server_port)
        # are fixed at construction in _start_with_handler and cannot be reassigned.
        inv.content.input_messages = self.input_messages
        inv.content.output_messages = self.output_messages
        inv.content.system_instruction = self.system_instruction
        inv.data.response_model = self.response_model_name
        inv.data.response_id = self.response_id
        inv.data.finish_reasons = self.finish_reasons
        inv.data.input_tokens = self.input_tokens
        inv.data.output_tokens = self.output_tokens

        inv.data.temperature = self.temperature
        inv.data.top_p = self.top_p
        inv.data.frequency_penalty = self.frequency_penalty
        inv.data.presence_penalty = self.presence_penalty
        inv.data.max_tokens = self.max_tokens
        inv.data.stop_sequences = self.stop_sequences
        inv.attributes.clear()
        inv.attributes.update(self.attributes)
        inv.data.attributes = inv.attributes
        inv.metric_attributes.clear()
        inv.metric_attributes.update(self.metric_attributes)
        inv.data.metric_attributes = inv.metric_attributes

    @property
    def span(self) -> Span:
        """The underlying span, for back-compat with code that checks span.is_recording()."""
        return (
            self._inference_invocation.span
            if self._inference_invocation is not None
            else INVALID_SPAN
        )
