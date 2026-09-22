# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.trace import INVALID_SPAN, Span, Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.semconv.gen_ai import (
    GenAiOperationName,
    GenAiTokenType,
)
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    InferenceClientOperation,
)
from opentelemetry.util.genai.types import (
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    SystemInstructionPart,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    _should_emit_event,
    get_content_capturing_mode,
)
from opentelemetry.util.types import AttributeValue

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


class InferenceInvocation(GenAIInvocation, InferenceClientOperation):
    """Represents a single LLM chat/completion call.

    Use handler.inference(provider) rather than constructing this directly.
    """

    system_instruction = _Attribute[
        list[SystemInstructionPart] | list[MessagePart] | None
    ]("system_instructions")
    """System instructions for the model. Passing ``MessagePart`` is deprecated; use ``SystemInstructionPart``."""
    response_model_name = _Attribute[str | None]("response_model")
    finish_reasons = _Attribute[list[str] | None]("response_finish_reasons")
    input_tokens = _Attribute[int | None]("usage_input_tokens")
    output_tokens = _Attribute[int | None]("usage_output_tokens")
    thinking_tokens = _Attribute[int | None]("usage_reasoning_output_tokens")
    temperature = _Attribute[float | None]("request_temperature")
    top_p = _Attribute[float | None]("request_top_p")
    top_k = _Attribute[int | None]("request_top_k")
    frequency_penalty = _Attribute[float | None]("request_frequency_penalty")
    presence_penalty = _Attribute[float | None]("request_presence_penalty")
    max_tokens = _Attribute[int | None]("request_max_tokens")
    stop_sequences = _Attribute[list[str] | None]("request_stop_sequences")
    seed = _Attribute[int | None]("request_seed")
    reasoning_level = _Attribute[str | None]("request_reasoning_level")
    previous_response_id = _Attribute[str | None](
        "request_previous_response_id"
    )
    cache_write_input_tokens = _Attribute[int | None](
        "usage_cache_write_input_tokens"
    )
    cache_read_input_tokens = _Attribute[int | None](
        "usage_cache_read_input_tokens"
    )
    text_input_tokens = _Attribute[int | None]("usage_text_input_tokens")
    image_input_tokens = _Attribute[int | None]("usage_image_input_tokens")
    audio_input_tokens = _Attribute[int | None]("usage_audio_input_tokens")
    text_output_tokens = _Attribute[int | None]("usage_text_output_tokens")
    image_output_tokens = _Attribute[int | None]("usage_image_output_tokens")
    audio_output_tokens = _Attribute[int | None]("usage_audio_output_tokens")
    text_cache_read_input_tokens = _Attribute[int | None](
        "usage_text_cache_read_input_tokens"
    )
    image_cache_read_input_tokens = _Attribute[int | None](
        "usage_image_cache_read_input_tokens"
    )
    audio_cache_read_input_tokens = _Attribute[int | None](
        "usage_audio_cache_read_input_tokens"
    )
    prompt_variables = _Attribute[Mapping[str, object] | None](
        "prompt_variable"
    )

    _request_stream = _Attribute[bool | None]("request_stream")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
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
        emit_event: bool | None = None,
    ) -> None:
        operation_name = operation_name or GenAiOperationName.CHAT.value
        """Use handler.inference(provider) rather than calling this directly."""
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        should_emit = (
            _should_emit_event(mode) if emit_event is None else emit_event
        )
        InferenceClientOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            completion_hook=completion_hook,
            error_type_resolver=error_type_resolver,
            operation_name=operation_name,
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=mode,
            emit_event=should_emit,
        )
        GenAIInvocation.__init__(self)
        self._emit_event: bool = self.emit_event
        self._stream_last_chunk_at: float | None = None
        self.start()

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
            setattr(self, field_name, None)
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
                setattr(self, field_name, token_count)

    @property
    def cache_creation_input_tokens(self) -> int | None:
        """
        .. deprecated:: 1.3b0
            Use :attr:`cache_write_input_tokens` instead.
        """
        return self.cache_write_input_tokens

    @cache_creation_input_tokens.setter
    def cache_creation_input_tokens(self, value: int | None) -> None:
        self.cache_write_input_tokens = value

    def record_stream_chunk(self) -> None:
        """Mark the request as streamed and record one output chunk arriving."""
        if self._context_token is None:
            return
        self._request_stream = True
        self._on_stream_chunk(timeit.default_timer())

    def _on_stream_chunk(self, chunk_at: float) -> None:
        is_first_chunk = self._stream_last_chunk_at is None
        last_chunk_at = (
            self._stream_last_chunk_at
            if self._stream_last_chunk_at is not None
            else self._monotonic_start_s
        )
        self._stream_last_chunk_at = chunk_at
        delta = max(chunk_at - last_chunk_at, 0.0)

        if is_first_chunk:
            if self.response_time_to_first_chunk is None:
                self.response_time_to_first_chunk = delta
            self.record_time_to_first_chunk(delta)
            return

        self.record_time_per_output_chunk(delta)

    def _on_finish(self, context: Context | None = None) -> None:
        if self.usage_input_tokens is not None:
            self.record_token_usage(
                self.usage_input_tokens,
                token_type=GenAiTokenType.INPUT,
                context=context,
            )
        if self.usage_output_tokens is not None:
            self.record_token_usage(
                self.usage_output_tokens,
                token_type=GenAiTokenType.OUTPUT,
                context=context,
            )


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
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
    ) -> None:
        """Create and start an InferenceInvocation from this data container. Called by handler.start_llm()."""
        inv = InferenceInvocation(
            tracer,
            meter,
            logger,
            completion_hook,
            self.provider or "",
            request_model=self.request_model,
            server_address=self.server_address,
            server_port=self.server_port,
            content_capturing_mode=content_capturing_mode,
            emit_event=emit_event,
        )
        inv.input_messages = self.input_messages
        inv.output_messages = self.output_messages
        inv.system_instruction = self.system_instruction
        inv.response_model_name = self.response_model_name
        inv.response_id = self.response_id
        inv.finish_reasons = self.finish_reasons
        inv.input_tokens = self.input_tokens
        inv.output_tokens = self.output_tokens

        inv.temperature = self.temperature
        inv.top_p = self.top_p
        inv.frequency_penalty = self.frequency_penalty
        inv.presence_penalty = self.presence_penalty
        inv.max_tokens = self.max_tokens
        inv.stop_sequences = self.stop_sequences
        inv.seed = self.seed
        inv.attributes.update(self.attributes)
        inv.metric_attributes.update(self.metric_attributes)
        self._inference_invocation = inv

    def _sync_to_invocation(self) -> None:
        inv = self._inference_invocation
        if inv is None:
            return
        # Start attributes (provider, request_model, server_address, server_port)
        # are fixed at construction in _start_with_handler and cannot be reassigned.
        inv.input_messages = self.input_messages
        inv.output_messages = self.output_messages
        inv.system_instruction = self.system_instruction
        inv.response_model_name = self.response_model_name
        inv.response_id = self.response_id
        inv.finish_reasons = self.finish_reasons
        inv.input_tokens = self.input_tokens
        inv.output_tokens = self.output_tokens

        inv.temperature = self.temperature
        inv.top_p = self.top_p
        inv.frequency_penalty = self.frequency_penalty
        inv.presence_penalty = self.presence_penalty
        inv.max_tokens = self.max_tokens
        inv.stop_sequences = self.stop_sequences
        inv.seed = self.seed
        inv.attributes = self.attributes
        inv.metric_attributes = self.metric_attributes

    @property
    def span(self) -> Span:
        """The underlying span, for back-compat with code that checks span.is_recording()."""
        return (
            self._inference_invocation.span
            if self._inference_invocation is not None
            else INVALID_SPAN
        )
