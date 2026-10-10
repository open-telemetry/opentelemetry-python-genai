# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Final

from opentelemetry._logs import Logger, LogRecord, SeverityNumber
from opentelemetry.context import Context, get_value
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import (
    SpanKind,
    Tracer,
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
    ContentCapturingMode,
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
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
    Modality.TEXT: "usage_text_input_tokens",
    Modality.IMAGE: "usage_image_input_tokens",
    Modality.AUDIO: "usage_audio_input_tokens",
}
_OUTPUT_MODALITY_FIELDS: Final[Mapping[str, str]] = {
    Modality.TEXT: "usage_text_output_tokens",
    Modality.IMAGE: "usage_image_output_tokens",
    Modality.AUDIO: "usage_audio_output_tokens",
}
_CACHE_READ_MODALITY_FIELDS: Final[Mapping[str, str]] = {
    Modality.TEXT: "usage_text_cache_read_input_tokens",
    Modality.IMAGE: "usage_image_cache_read_input_tokens",
    Modality.AUDIO: "usage_audio_cache_read_input_tokens",
}
_GEN_AI_REQUEST_REASONING_LEVEL: Final = "gen_ai.request.reasoning.level"
_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: Final = (
    "gen_ai.request.previous_response.id"
)
_GEN_AI_CONVERSATION_COMPACTED: Final = "gen_ai.conversation.compacted"
_GEN_AI_PROMPT_VERSION: Final = "gen_ai.prompt.version"


def _should_emit_event(
    content_capturing_mode: ContentCapturingMode,
) -> bool:
    """Check if event emission is enabled."""
    return content_capturing_mode in (
        ContentCapturingMode.EVENT_ONLY,
        ContentCapturingMode.SPAN_AND_EVENT,
    )


CLIENT_INFERENCE_CONTEXT_KEY: Final[str] = (
    "opentelemetry.genai.client.inference.context"
)


@dataclass
class InferenceData:
    """Typed data passed from inner inference invocations to the outer invocation."""

    conversation_id: str | None = None
    conversation_compacted: bool | None = None
    error_type: str | None = None
    input_messages: Sequence[InputMessage] | None = None
    operation_name: str | None = None
    output_messages: Sequence[OutputMessage] | None = None
    output_type: str | None = None
    prompt_name: str | None = None
    prompt_variable: Mapping[str, object] | None = None
    prompt_version: str | None = None
    provider_name: str | None = None
    request_choice_count: int | None = None
    request_frequency_penalty: float | None = None
    request_max_tokens: int | None = None
    request_model: str | None = None
    request_presence_penalty: float | None = None
    request_previous_response_id: str | None = None
    request_reasoning_level: str | None = None
    request_seed: int | None = None
    request_stop_sequences: list[str] | None = None
    request_stream: bool | None = None
    request_temperature: float | None = None
    request_top_k: int | None = None
    request_top_p: float | None = None
    response_finish_reasons: list[str] | None = None
    response_id: str | None = None
    response_model: str | None = None
    response_time_to_first_chunk: float | None = None
    server_address: str | None = None
    server_port: int | None = None
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
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )

    def merge(self, other: InferenceData, *, overwrite: bool = True) -> None:
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


class InferenceInvocation(GenAIInvocation):
    """Represents a single LLM chat/completion call.

    Use handler.inference(provider) rather than constructing this directly.
    """

    _context_key = CLIENT_INFERENCE_CONTEXT_KEY
    _dataclass_class_object = InferenceData

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
        self.data: InferenceData = InferenceData(
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            conversation_id=conversation_id,
            operation_name=operation_name,
            input_messages=[],
            output_messages=[],
            system_instructions=[],
        )
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
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
        )
        self._provider: str = provider
        self._request_model: str | None = request_model
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        self._emit_event: bool = _should_emit_event(
            self._content_capturing_mode
        )
        if (
            self._emit_event
            and isinstance(self._completion_hook, _NoOpCompletionHook)
            and hasattr(self._logger, "enabled")
        ):
            self._emit_event = self._logger.enabled(
                context=self._span_context,
                severity_number=SeverityNumber.DEBUG,
                event_name="gen_ai.client.inference.operation.details",
            )

    @property
    def provider(self) -> str | None:
        return self.data.provider_name

    @property
    def request_model(self) -> str | None:
        return self.data.request_model

    @property
    def response_model_name(self) -> str | None:
        return self.data.response_model

    @response_model_name.setter
    def response_model_name(self, value: str | None) -> None:
        self.data.response_model = value

    @property
    def server_address(self) -> str | None:
        return self.data.server_address

    @property
    def server_port(self) -> int | None:
        return self.data.server_port

    @property
    def response_id(self) -> str | None:
        return self.data.response_id

    @response_id.setter
    def response_id(self, value: str | None) -> None:
        self.data.response_id = value

    @property
    def finish_reasons(self) -> list[str] | None:
        return self.data.response_finish_reasons

    @finish_reasons.setter
    def finish_reasons(self, value: Sequence[str] | None) -> None:
        self.data.response_finish_reasons = (
            list(value) if value is not None else None
        )

    @property
    def input_tokens(self) -> int | None:
        return self.data.usage_input_tokens

    @input_tokens.setter
    def input_tokens(self, value: int | None) -> None:
        self.data.usage_input_tokens = value

    @property
    def output_tokens(self) -> int | None:
        return self.data.usage_output_tokens

    @output_tokens.setter
    def output_tokens(self, value: int | None) -> None:
        self.data.usage_output_tokens = value

    @property
    def thinking_tokens(self) -> int | None:
        return self.data.usage_reasoning_output_tokens

    @thinking_tokens.setter
    def thinking_tokens(self, value: int | None) -> None:
        self.data.usage_reasoning_output_tokens = value

    @property
    def temperature(self) -> float | None:
        return self.data.request_temperature

    @temperature.setter
    def temperature(self, value: float | None) -> None:
        self.data.request_temperature = value

    @property
    def top_p(self) -> float | None:
        return self.data.request_top_p

    @top_p.setter
    def top_p(self, value: float | None) -> None:
        self.data.request_top_p = value

    @property
    def top_k(self) -> int | None:
        return self.data.request_top_k

    @top_k.setter
    def top_k(self, value: int | None) -> None:
        self.data.request_top_k = value

    @property
    def frequency_penalty(self) -> float | None:
        return self.data.request_frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float | None) -> None:
        self.data.request_frequency_penalty = value

    @property
    def presence_penalty(self) -> float | None:
        return self.data.request_presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float | None) -> None:
        self.data.request_presence_penalty = value

    @property
    def max_tokens(self) -> int | None:
        return self.data.request_max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None) -> None:
        self.data.request_max_tokens = value

    @property
    def stop_sequences(self) -> list[str] | None:
        return self.data.request_stop_sequences

    @stop_sequences.setter
    def stop_sequences(self, value: Sequence[str] | None) -> None:
        self.data.request_stop_sequences = (
            list(value) if value is not None else None
        )

    @property
    def seed(self) -> int | None:
        return self.data.request_seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        self.data.request_seed = value

    @property
    def request_choice_count(self) -> int | None:
        return self.data.request_choice_count

    @request_choice_count.setter
    def request_choice_count(self, value: int | None) -> None:
        self.data.request_choice_count = value

    @property
    def output_type(self) -> str | None:
        return self.data.output_type

    @output_type.setter
    def output_type(self, value: str | None) -> None:
        self.data.output_type = value

    @property
    def request_stream(self) -> bool | None:
        return (
            self._request_stream
            if self._request_stream is not None
            else self.data.request_stream
        )

    @request_stream.setter
    def request_stream(self, value: bool | None) -> None:
        self._request_stream = value
        self.data.request_stream = value

    @property
    def ttfc_seconds(self) -> float | None:
        return self.data.response_time_to_first_chunk

    @ttfc_seconds.setter
    def ttfc_seconds(self, value: float | None) -> None:
        self.data.response_time_to_first_chunk = value

    @property
    def cache_write_input_tokens(self) -> int | None:
        return self.data.usage_cache_write_input_tokens

    @cache_write_input_tokens.setter
    def cache_write_input_tokens(self, value: int | None) -> None:
        self.data.usage_cache_write_input_tokens = value

    @property
    def cache_creation_input_tokens(self) -> int | None:
        """
        .. deprecated:: 1.3b0
            Use :attr:`cache_write_input_tokens` instead.
        """
        return self.data.usage_cache_write_input_tokens

    @cache_creation_input_tokens.setter
    def cache_creation_input_tokens(self, value: int | None) -> None:
        self.data.usage_cache_write_input_tokens = value

    @property
    def cache_read_input_tokens(self) -> int | None:
        return self.data.usage_cache_read_input_tokens

    @cache_read_input_tokens.setter
    def cache_read_input_tokens(self, value: int | None) -> None:
        self.data.usage_cache_read_input_tokens = value

    @property
    def text_input_tokens(self) -> int | None:
        return self.data.usage_text_input_tokens

    @text_input_tokens.setter
    def text_input_tokens(self, value: int | None) -> None:
        self.data.usage_text_input_tokens = value

    @property
    def image_input_tokens(self) -> int | None:
        return self.data.usage_image_input_tokens

    @image_input_tokens.setter
    def image_input_tokens(self, value: int | None) -> None:
        self.data.usage_image_input_tokens = value

    @property
    def audio_input_tokens(self) -> int | None:
        return self.data.usage_audio_input_tokens

    @audio_input_tokens.setter
    def audio_input_tokens(self, value: int | None) -> None:
        self.data.usage_audio_input_tokens = value

    @property
    def text_output_tokens(self) -> int | None:
        return self.data.usage_text_output_tokens

    @text_output_tokens.setter
    def text_output_tokens(self, value: int | None) -> None:
        self.data.usage_text_output_tokens = value

    @property
    def image_output_tokens(self) -> int | None:
        return self.data.usage_image_output_tokens

    @image_output_tokens.setter
    def image_output_tokens(self, value: int | None) -> None:
        self.data.usage_image_output_tokens = value

    @property
    def audio_output_tokens(self) -> int | None:
        return self.data.usage_audio_output_tokens

    @audio_output_tokens.setter
    def audio_output_tokens(self, value: int | None) -> None:
        self.data.usage_audio_output_tokens = value

    @property
    def text_cache_read_input_tokens(self) -> int | None:
        return self.data.usage_text_cache_read_input_tokens

    @text_cache_read_input_tokens.setter
    def text_cache_read_input_tokens(self, value: int | None) -> None:
        self.data.usage_text_cache_read_input_tokens = value

    @property
    def image_cache_read_input_tokens(self) -> int | None:
        return self.data.usage_image_cache_read_input_tokens

    @image_cache_read_input_tokens.setter
    def image_cache_read_input_tokens(self, value: int | None) -> None:
        self.data.usage_image_cache_read_input_tokens = value

    @property
    def audio_cache_read_input_tokens(self) -> int | None:
        return self.data.usage_audio_cache_read_input_tokens

    @audio_cache_read_input_tokens.setter
    def audio_cache_read_input_tokens(self, value: int | None) -> None:
        self.data.usage_audio_cache_read_input_tokens = value

    @property
    def reasoning_level(self) -> str | None:
        return self.data.request_reasoning_level

    @reasoning_level.setter
    def reasoning_level(self, value: str | None) -> None:
        self.data.request_reasoning_level = value

    @property
    def previous_response_id(self) -> str | None:
        return self.data.request_previous_response_id

    @previous_response_id.setter
    def previous_response_id(self, value: str | None) -> None:
        self.data.request_previous_response_id = value

    @property
    def conversation_compacted(self) -> bool | None:
        return self.data.conversation_compacted

    @conversation_compacted.setter
    def conversation_compacted(self, value: bool | None) -> None:
        self.data.conversation_compacted = value

    @property
    def prompt_name(self) -> str | None:
        return self.data.prompt_name

    @prompt_name.setter
    def prompt_name(self, value: str | None) -> None:
        self.data.prompt_name = value

    @property
    def prompt_version(self) -> str | None:
        return self.data.prompt_version

    @prompt_version.setter
    def prompt_version(self, value: str | None) -> None:
        self.data.prompt_version = value

    @property
    def input_messages(self) -> list[InputMessage]:
        if isinstance(self.data.input_messages, list):
            return self.data.input_messages
        messages = (
            list(self.data.input_messages) if self.data.input_messages else []
        )
        self.data.input_messages = messages
        return messages

    @input_messages.setter
    def input_messages(self, value: Sequence[InputMessage]) -> None:
        self.data.input_messages = value

    @property
    def output_messages(self) -> list[OutputMessage]:
        if isinstance(self.data.output_messages, list):
            return self.data.output_messages
        messages = (
            list(self.data.output_messages)
            if self.data.output_messages
            else []
        )
        self.data.output_messages = messages
        return messages

    @output_messages.setter
    def output_messages(self, value: Sequence[OutputMessage]) -> None:
        self.data.output_messages = value

    @property
    def system_instruction(
        self,
    ) -> list[SystemInstructionPart] | list[MessagePart]:
        """System instructions for the model. Passing ``MessagePart`` is deprecated; use ``SystemInstructionPart``."""
        if isinstance(self.data.system_instructions, list):
            return self.data.system_instructions
        instructions = (
            list(self.data.system_instructions)
            if self.data.system_instructions
            else []
        )
        self.data.system_instructions = instructions
        return instructions

    @system_instruction.setter
    def system_instruction(
        self,
        value: Sequence[SystemInstructionPart] | Sequence[MessagePart] | None,
    ) -> None:
        self.data.system_instructions = (
            list(value) if value is not None else None
        )

    @property
    def prompt_variables(self) -> Mapping[str, object] | None:
        return self.data.prompt_variable

    @prompt_variables.setter
    def prompt_variables(self, value: Mapping[str, object] | None) -> None:
        self.data.prompt_variable = value

    @property
    def tool_definitions(self) -> list[ToolDefinition] | None:
        return self.data.tool_definitions

    @tool_definitions.setter
    def tool_definitions(self, value: Sequence[ToolDefinition] | None) -> None:
        self.data.tool_definitions = list(value) if value is not None else None

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
            input_messages=self.input_messages,
            output_messages=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
            prompt_variables=self.prompt_variables,
            for_span=for_span,
            content_capturing_mode=self._content_capturing_mode,
        )

    def _get_finish_reasons(self) -> list[str] | None:
        if self.data.response_finish_reasons is not None:
            return list(self.data.response_finish_reasons) or None
        if self.data.output_messages:
            reasons = [
                msg.finish_reason
                for msg in self.data.output_messages
                if msg.finish_reason
            ]
            return reasons or None
        return None

    def _on_stream_chunk(self, chunk_at: float) -> None:
        super()._on_stream_chunk(chunk_at)
        self.data.request_stream = True
        self.data.response_time_to_first_chunk = self._ttfc_seconds

    def _get_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        optional_attrs = (
            (GenAI.GEN_AI_PROVIDER_NAME, self.data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, self.data.request_model),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
            (
                GenAI.GEN_AI_CONVERSATION_ID,
                self.conversation_id or self.data.conversation_id,
            ),
            (GenAI.GEN_AI_REQUEST_STREAM, self.request_stream),
            (GenAI.GEN_AI_REQUEST_TEMPERATURE, self.data.request_temperature),
            (GenAI.GEN_AI_REQUEST_TOP_P, self.data.request_top_p),
            (GenAI.GEN_AI_REQUEST_TOP_K, self.data.request_top_k),
            (
                GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                self.data.request_frequency_penalty,
            ),
            (
                GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY,
                self.data.request_presence_penalty,
            ),
            (GenAI.GEN_AI_REQUEST_MAX_TOKENS, self.data.request_max_tokens),
            (
                GenAI.GEN_AI_REQUEST_STOP_SEQUENCES,
                self.data.request_stop_sequences,
            ),
            (GenAI.GEN_AI_REQUEST_SEED, self.data.request_seed),
            (GenAI.GEN_AI_RESPONSE_FINISH_REASONS, self._get_finish_reasons()),
            (GenAI.GEN_AI_RESPONSE_MODEL, self.data.response_model),
            (GenAI.GEN_AI_RESPONSE_ID, self.data.response_id),
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, self.data.usage_input_tokens),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, self.data.usage_output_tokens),
            (
                GenAI.GEN_AI_REQUEST_CHOICE_COUNT,
                self.data.request_choice_count,
            ),
            (GenAI.GEN_AI_OUTPUT_TYPE, self.data.output_type),
            (
                _GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
                self.data.usage_cache_write_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                self.data.usage_cache_read_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                self.data.usage_reasoning_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_INPUT_TOKENS,
                self.data.usage_text_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
                self.data.usage_image_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
                self.data.usage_audio_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
                self.data.usage_text_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
                self.data.usage_image_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
                self.data.usage_audio_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
                self.data.usage_text_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
                self.data.usage_image_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
                self.data.usage_audio_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_REQUEST_REASONING_LEVEL,
                self.data.request_reasoning_level,
            ),
            (
                _GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
                self.data.request_previous_response_id,
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
                self.data.response_time_to_first_chunk,
            ),
        )
        attrs.update({k: v for k, v in optional_attrs if v is not None})
        return attrs

    def enrich_from_context(self, data: InferenceData) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values. Inner
        invocations never override content capture fields.
        """
        # Content capture is always set to false on inner invocations.
        # We always want to use content capture from the outer invocation.
        input_messages = self.data.input_messages
        output_messages = self.data.output_messages
        system_instructions = self.data.system_instructions
        prompt_variable = self.data.prompt_variable
        tool_definitions = self.data.tool_definitions

        self.data.merge(data, overwrite=False)

        self.data.input_messages = input_messages
        self.data.output_messages = output_messages
        self.data.system_instructions = system_instructions
        self.data.prompt_variable = prompt_variable
        self.data.tool_definitions = tool_definitions

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs = dict(self._start_attributes)
        if self.data.provider_name is not None:
            attrs[GenAI.GEN_AI_PROVIDER_NAME] = self.data.provider_name
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
        if self.data.usage_input_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.INPUT.value] = (
                self.data.usage_input_tokens
            )
        if self.data.usage_output_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.OUTPUT.value] = (
                self.data.usage_output_tokens
            )
        return counts

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_value(
            CLIENT_INFERENCE_CONTEXT_KEY, context=self._span_context
        )
        if isinstance(ctx_data, InferenceData):
            self.enrich_from_context(ctx_data)
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        if self._request_stream is not None:
            self.data.request_stream = self._request_stream
        attributes: dict[str, AttributeValue] = {}
        attributes.update(self._get_attributes())
        attributes.update(self._get_message_attributes(for_span=True))
        attributes.update(self.attributes)
        start_keys = set(self._start_attributes)
        span_attributes = {
            k: v for k, v in attributes.items() if k not in start_keys
        }
        self.span.set_attributes(span_attributes)
        self._record_client_metrics()
        log_record = self._maybe_create_event()
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
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
            severity_number=SeverityNumber.DEBUG,
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
            self.data.response_time_to_first_chunk = delta
        self.data.request_stream = True

    def publish_to_context(self, data: InferenceData) -> None:
        """Publish invocation attributes to the active inference context."""
        # Sync fields managed on base GenAIInvocation onto self.data before merging.
        self.data.conversation_id = self.conversation_id
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        if self._request_stream is not None:
            self.data.request_stream = self._request_stream
        # Overwrite so later inner invocations update the context; the root invocation enriches with overwrite=False.
        data.merge(self.data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True

        if self.data.response_finish_reasons is None:
            self.data.response_finish_reasons = self._get_finish_reasons()

        ctx_data = get_value(
            CLIENT_INFERENCE_CONTEXT_KEY, context=self._span_context
        )
        if isinstance(ctx_data, InferenceData):
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass


@dataclass
class LLMInvocation:
    """Deprecated compatibility data container for an LLM invocation.

    Use ``handler.inference()`` to create an ``InferenceInvocation`` instead.
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
