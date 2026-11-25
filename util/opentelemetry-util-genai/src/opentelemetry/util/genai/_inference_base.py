# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Final, Protocol

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    Modality,
    ModalityTokens,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

__all__ = [
    "_BaseInferenceInvocation",
    "_InferenceData",
]

_GEN_AI_REQUEST_REASONING_LEVEL: Final = "gen_ai.request.reasoning.level"
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


class _InferenceData(Protocol):
    """Typed structural contract shared by inference and realtime data containers."""

    input_messages: Sequence[InputMessage] | None
    output_messages: Sequence[OutputMessage] | None
    output_type: str | None
    provider_name: str | None
    request_max_tokens: int | None
    request_model: str | None
    request_reasoning_level: str | None
    response_finish_reasons: list[str] | None
    response_id: str | None
    response_model: str | None
    server_address: str | None
    server_port: int | None
    system_instructions: (
        Sequence[SystemInstructionPart] | Sequence[MessagePart] | None
    )
    tool_definitions: list[ToolDefinition] | None
    usage_audio_cache_read_input_tokens: int | None
    usage_audio_input_tokens: int | None
    usage_audio_output_tokens: int | None
    usage_cache_read_input_tokens: int | None
    usage_cache_write_input_tokens: int | None
    usage_image_cache_read_input_tokens: int | None
    usage_image_input_tokens: int | None
    usage_image_output_tokens: int | None
    usage_input_tokens: int | None
    usage_output_tokens: int | None
    usage_reasoning_output_tokens: int | None
    usage_text_cache_read_input_tokens: int | None
    usage_text_input_tokens: int | None
    usage_text_output_tokens: int | None


def _parse_modality_tokens(
    entries: ModalityTokens,
) -> tuple[int | None, int | None, int | None]:
    """Parse ``(modality, count)`` pairs into ``(text, image, audio)`` token counts."""
    text_tokens: int | None = None
    image_tokens: int | None = None
    audio_tokens: int | None = None
    for modality, token_count in entries:
        if (
            not isinstance(token_count, int)
            or isinstance(token_count, bool)
            or token_count < 0
        ):
            continue
        # modality may be an enum, whose str() is "MediaModality.AUDIO"
        # rather than the bare name.
        key = str(getattr(modality, "value", modality)).lower()
        if key == Modality.TEXT:
            text_tokens = token_count
        elif key == Modality.IMAGE:
            image_tokens = token_count
        elif key == Modality.AUDIO:
            audio_tokens = token_count
    return text_tokens, image_tokens, audio_tokens


class _BaseInferenceInvocation(GenAIInvocation):
    """Shared typed accessors and token helpers for inference invocation types."""

    @property
    @abstractmethod
    def _inference_data(self) -> _InferenceData:
        """Return the typed data container backing shared inference accessors."""

    @property
    def provider(self) -> str | None:
        return self._inference_data.provider_name

    @property
    def request_model(self) -> str | None:
        return self._inference_data.request_model

    @property
    def response_model_name(self) -> str | None:
        return self._inference_data.response_model

    @response_model_name.setter
    def response_model_name(self, value: str | None) -> None:
        self._inference_data.response_model = value

    @property
    def server_address(self) -> str | None:
        return self._inference_data.server_address

    @property
    def server_port(self) -> int | None:
        return self._inference_data.server_port

    @property
    def response_id(self) -> str | None:
        return self._inference_data.response_id

    @response_id.setter
    def response_id(self, value: str | None) -> None:
        self._inference_data.response_id = value

    @property
    def finish_reasons(self) -> list[str] | None:
        return self._inference_data.response_finish_reasons

    @finish_reasons.setter
    def finish_reasons(self, value: Sequence[str] | None) -> None:
        self._inference_data.response_finish_reasons = (
            list(value) if value is not None else None
        )

    @property
    def max_tokens(self) -> int | None:
        return self._inference_data.request_max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None) -> None:
        self._inference_data.request_max_tokens = value

    @property
    def output_type(self) -> str | None:
        return self._inference_data.output_type

    @output_type.setter
    def output_type(self, value: str | None) -> None:
        self._inference_data.output_type = value

    @property
    def reasoning_level(self) -> str | None:
        return self._inference_data.request_reasoning_level

    @reasoning_level.setter
    def reasoning_level(self, value: str | None) -> None:
        self._inference_data.request_reasoning_level = value

    @property
    def input_tokens(self) -> int | None:
        return self._inference_data.usage_input_tokens

    @input_tokens.setter
    def input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_input_tokens = value

    @property
    def output_tokens(self) -> int | None:
        return self._inference_data.usage_output_tokens

    @output_tokens.setter
    def output_tokens(self, value: int | None) -> None:
        self._inference_data.usage_output_tokens = value

    @property
    def thinking_tokens(self) -> int | None:
        return self._inference_data.usage_reasoning_output_tokens

    @thinking_tokens.setter
    def thinking_tokens(self, value: int | None) -> None:
        self._inference_data.usage_reasoning_output_tokens = value

    @property
    def cache_write_input_tokens(self) -> int | None:
        return self._inference_data.usage_cache_write_input_tokens

    @cache_write_input_tokens.setter
    def cache_write_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_cache_write_input_tokens = value

    @property
    def cache_read_input_tokens(self) -> int | None:
        return self._inference_data.usage_cache_read_input_tokens

    @cache_read_input_tokens.setter
    def cache_read_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_cache_read_input_tokens = value

    @property
    def text_input_tokens(self) -> int | None:
        return self._inference_data.usage_text_input_tokens

    @text_input_tokens.setter
    def text_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_text_input_tokens = value

    @property
    def image_input_tokens(self) -> int | None:
        return self._inference_data.usage_image_input_tokens

    @image_input_tokens.setter
    def image_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_image_input_tokens = value

    @property
    def audio_input_tokens(self) -> int | None:
        return self._inference_data.usage_audio_input_tokens

    @audio_input_tokens.setter
    def audio_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_audio_input_tokens = value

    @property
    def text_output_tokens(self) -> int | None:
        return self._inference_data.usage_text_output_tokens

    @text_output_tokens.setter
    def text_output_tokens(self, value: int | None) -> None:
        self._inference_data.usage_text_output_tokens = value

    @property
    def image_output_tokens(self) -> int | None:
        return self._inference_data.usage_image_output_tokens

    @image_output_tokens.setter
    def image_output_tokens(self, value: int | None) -> None:
        self._inference_data.usage_image_output_tokens = value

    @property
    def audio_output_tokens(self) -> int | None:
        return self._inference_data.usage_audio_output_tokens

    @audio_output_tokens.setter
    def audio_output_tokens(self, value: int | None) -> None:
        self._inference_data.usage_audio_output_tokens = value

    @property
    def text_cache_read_input_tokens(self) -> int | None:
        return self._inference_data.usage_text_cache_read_input_tokens

    @text_cache_read_input_tokens.setter
    def text_cache_read_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_text_cache_read_input_tokens = value

    @property
    def image_cache_read_input_tokens(self) -> int | None:
        return self._inference_data.usage_image_cache_read_input_tokens

    @image_cache_read_input_tokens.setter
    def image_cache_read_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_image_cache_read_input_tokens = value

    @property
    def audio_cache_read_input_tokens(self) -> int | None:
        return self._inference_data.usage_audio_cache_read_input_tokens

    @audio_cache_read_input_tokens.setter
    def audio_cache_read_input_tokens(self, value: int | None) -> None:
        self._inference_data.usage_audio_cache_read_input_tokens = value

    @property
    def input_messages(self) -> list[InputMessage]:
        data = self._inference_data
        if isinstance(data.input_messages, list):
            return data.input_messages
        messages = list(data.input_messages) if data.input_messages else []
        data.input_messages = messages
        return messages

    @input_messages.setter
    def input_messages(self, value: Sequence[InputMessage]) -> None:
        self._inference_data.input_messages = value

    @property
    def output_messages(self) -> list[OutputMessage]:
        data = self._inference_data
        if isinstance(data.output_messages, list):
            return data.output_messages
        messages = list(data.output_messages) if data.output_messages else []
        data.output_messages = messages
        return messages

    @output_messages.setter
    def output_messages(self, value: Sequence[OutputMessage]) -> None:
        self._inference_data.output_messages = value

    @property
    def system_instruction(
        self,
    ) -> list[SystemInstructionPart] | list[MessagePart]:
        """System instructions for the model. Passing ``MessagePart`` is deprecated; use ``SystemInstructionPart``."""
        data = self._inference_data
        if isinstance(data.system_instructions, list):
            return data.system_instructions
        instructions = (
            list(data.system_instructions) if data.system_instructions else []
        )
        data.system_instructions = instructions
        return instructions

    @system_instruction.setter
    def system_instruction(
        self,
        value: Sequence[SystemInstructionPart] | Sequence[MessagePart] | None,
    ) -> None:
        self._inference_data.system_instructions = (
            list(value) if value is not None else None
        )

    @property
    def tool_definitions(self) -> list[ToolDefinition] | None:
        return self._inference_data.tool_definitions

    @tool_definitions.setter
    def tool_definitions(self, value: Sequence[ToolDefinition] | None) -> None:
        self._inference_data.tool_definitions = (
            list(value) if value is not None else None
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
        if entries is None:
            return
        data = self._inference_data
        (
            data.usage_text_input_tokens,
            data.usage_image_input_tokens,
            data.usage_audio_input_tokens,
        ) = _parse_modality_tokens(entries)

    def set_output_tokens(self, entries: ModalityTokens | None) -> None:
        """Record the per-modality breakdown of the output tokens.

        Sets ``gen_ai.usage.{text,image,audio}.output_tokens``. See
        :meth:`set_input_tokens` for the argument contract.
        """
        if entries is None:
            return
        data = self._inference_data
        (
            data.usage_text_output_tokens,
            data.usage_image_output_tokens,
            data.usage_audio_output_tokens,
        ) = _parse_modality_tokens(entries)

    def set_cache_read_input_tokens(
        self, entries: ModalityTokens | None
    ) -> None:
        """Record the per-modality breakdown of the cache read input tokens.

        Sets ``gen_ai.usage.{text,image,audio}.cache_read.input_tokens``. See
        :meth:`set_input_tokens` for the argument contract.
        """
        if entries is None:
            return
        data = self._inference_data
        (
            data.usage_text_cache_read_input_tokens,
            data.usage_image_cache_read_input_tokens,
            data.usage_audio_cache_read_input_tokens,
        ) = _parse_modality_tokens(entries)

    def _get_finish_reasons(self) -> list[str] | None:
        """Return explicit finish reasons, falling back to deprecated message-level values."""
        data = self._inference_data
        if data.response_finish_reasons is not None:
            return list(data.response_finish_reasons) or None
        if data.output_messages:
            reasons = [
                msg.finish_reason
                for msg in data.output_messages
                if msg.finish_reason
            ]
            return reasons or None
        return None

    def _get_common_attributes(self) -> dict[str, AttributeValue]:
        """Return typed attributes shared by inference and realtime spans."""
        data = self._inference_data
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_PROVIDER_NAME, data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, data.request_model),
            (server_attributes.SERVER_ADDRESS, data.server_address),
            (server_attributes.SERVER_PORT, data.server_port),
            (GenAI.GEN_AI_OUTPUT_TYPE, data.output_type),
            (GenAI.GEN_AI_REQUEST_MAX_TOKENS, data.request_max_tokens),
            (
                _GEN_AI_REQUEST_REASONING_LEVEL,
                data.request_reasoning_level,
            ),
            (GenAI.GEN_AI_RESPONSE_FINISH_REASONS, self._get_finish_reasons()),
            (GenAI.GEN_AI_RESPONSE_ID, data.response_id),
            (GenAI.GEN_AI_RESPONSE_MODEL, data.response_model),
        )
        attrs: dict[str, AttributeValue] = {
            k: v for k, v in optional_attrs if v is not None
        }
        attrs.update(self._get_token_usage_attributes())
        return attrs

    def _get_token_usage_attributes(self) -> dict[str, AttributeValue]:
        """Return non-empty GenAI token usage attributes for span/event recording."""
        data = self._inference_data
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, data.usage_input_tokens),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, data.usage_output_tokens),
            (
                _GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
                data.usage_cache_write_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                data.usage_cache_read_input_tokens or None,
            ),
            (
                GenAI.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                data.usage_reasoning_output_tokens or None,
            ),
            (_GEN_AI_USAGE_TEXT_INPUT_TOKENS, data.usage_text_input_tokens or None),
            (
                _GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
                data.usage_image_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
                data.usage_audio_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
                data.usage_text_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
                data.usage_image_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
                data.usage_audio_output_tokens or None,
            ),
            (
                _GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
                data.usage_text_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
                data.usage_image_cache_read_input_tokens or None,
            ),
            (
                _GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
                data.usage_audio_cache_read_input_tokens or None,
            ),
        )
        return {k: v for k, v in optional_attrs if v is not None}
