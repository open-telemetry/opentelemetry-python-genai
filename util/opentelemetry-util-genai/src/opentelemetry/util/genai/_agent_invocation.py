# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
    _ContextData,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import ContentCapturingMode
from opentelemetry.util.types import AttributeValue

_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: Final = (
    "gen_ai.usage.cache_write.input_tokens"
)
_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: Final = (
    "gen_ai.request.previous_response.id"
)


@dataclass
class AgentData(_ContextData):
    """Typed data container for an agent invocation."""

    agent_id: str | None = None
    agent_name: str | None = None
    agent_description: str | None = None
    agent_version: str | None = None
    conversation_id: str | None = None
    data_source_id: str | None = None
    input_messages: Sequence[InputMessage] | None = None
    output_messages: Sequence[OutputMessage] | None = None
    output_type: str | None = None
    provider_name: str | None = None
    request_choice_count: int | None = None
    request_frequency_penalty: float | None = None
    request_max_tokens: int | None = None
    request_model: str | None = None
    request_presence_penalty: float | None = None
    request_previous_response_id: str | None = None
    request_seed: int | None = None
    request_stop_sequences: list[str] | None = None
    request_temperature: float | None = None
    request_top_p: float | None = None
    response_finish_reasons: list[str] | None = None
    server_address: str | None = None
    server_port: int | None = None
    system_instructions: (
        Sequence[SystemInstructionPart] | Sequence[MessagePart] | None
    ) = None
    tool_definitions: list[ToolDefinition] | None = None
    usage_cache_read_input_tokens: int | None = None
    usage_cache_write_input_tokens: int | None = None
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class AgentInvocation(GenAIInvocation, ABC):
    """Base class representing a GenAI agent invocation (invoke_agent span).

    Use handler.invoke_local_agent() or handler.invoke_remote_agent()
    rather than constructing this directly.

    Reference:
        Client span: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-client-span
        Internal span: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-internal-span
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        span_kind: SpanKind,
        start_attributes: dict[str, AttributeValue] | None = None,
        request_model: str | None = None,
        agent_name: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
        data: AgentData | None = None,
    ) -> None:
        _operation_name = GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
        if start_attributes is None:
            start_attributes = {
                k: v
                for k, v in (
                    (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                    (GenAI.GEN_AI_AGENT_NAME, agent_name),
                )
                if v is not None
            }
        if data is None:
            data = AgentData(
                agent_name=agent_name,
                request_model=request_model,
                conversation_id=conversation_id,
                input_messages=[],
                output_messages=[],
                system_instructions=[],
            )
        self.data: AgentData = data
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {agent_name}"
            if agent_name
            else _operation_name,
            span_kind=span_kind,
            start_attributes=start_attributes,
            context=context,
            conversation_id=conversation_id,
            content_capturing_mode=content_capturing_mode,
            _attach_to_context=_attach_to_context,
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
        )
        self._request_model: str | None = request_model
        self._agent_name: str | None = agent_name
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes

    @property
    def agent_name(self) -> str | None:
        """The agent name provided at construction time."""
        return self.data.agent_name

    @property
    def request_model(self) -> str | None:
        """The request model provided at construction time."""
        return self.data.request_model

    @property
    def agent_description(self) -> str | None:
        return self.data.agent_description

    @agent_description.setter
    def agent_description(self, value: str | None) -> None:
        self.data.agent_description = value

    @property
    def data_source_id(self) -> str | None:
        return self.data.data_source_id

    @data_source_id.setter
    def data_source_id(self, value: str | None) -> None:
        self.data.data_source_id = value

    @property
    def output_type(self) -> str | None:
        return self.data.output_type

    @output_type.setter
    def output_type(self, value: str | None) -> None:
        self.data.output_type = value

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
    def choice_count(self) -> int | None:
        return self.data.request_choice_count

    @choice_count.setter
    def choice_count(self, value: int | None) -> None:
        self.data.request_choice_count = value

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
        self, value: Sequence[SystemInstructionPart] | Sequence[MessagePart]
    ) -> None:
        self.data.system_instructions = value

    @property
    def tool_definitions(self) -> list[ToolDefinition] | None:
        return self.data.tool_definitions

    @tool_definitions.setter
    def tool_definitions(self, value: Sequence[ToolDefinition] | None) -> None:
        self.data.tool_definitions = list(value) if value is not None else None

    def _get_agent_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_AGENT_DESCRIPTION, self.agent_description),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _get_request_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_CONVERSATION_ID, self.conversation_id),
            (GenAI.GEN_AI_DATA_SOURCE_ID, self.data_source_id),
            (GenAI.GEN_AI_OUTPUT_TYPE, self.output_type),
            (GenAI.GEN_AI_REQUEST_TEMPERATURE, self.temperature),
            (GenAI.GEN_AI_REQUEST_TOP_P, self.top_p),
            (GenAI.GEN_AI_REQUEST_FREQUENCY_PENALTY, self.frequency_penalty),
            (GenAI.GEN_AI_REQUEST_PRESENCE_PENALTY, self.presence_penalty),
            (GenAI.GEN_AI_REQUEST_MAX_TOKENS, self.max_tokens),
            (GenAI.GEN_AI_REQUEST_STOP_SEQUENCES, self.stop_sequences),
            (GenAI.GEN_AI_REQUEST_SEED, self.seed),
            (GenAI.GEN_AI_REQUEST_CHOICE_COUNT, self.choice_count),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _get_response_attributes(self) -> dict[str, AttributeValue]:
        if self.finish_reasons:
            return {GenAI.GEN_AI_RESPONSE_FINISH_REASONS: self.finish_reasons}
        return {}

    def _get_usage_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, self.data.usage_input_tokens),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, self.data.usage_output_tokens),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _get_content_attributes_for_span(self) -> dict[str, AttributeValue]:
        return get_content_attributes(
            input_messages=self.input_messages,
            output_messages=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
            for_span=True,
            content_capturing_mode=self._content_capturing_mode,
        )

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)

        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        attributes: dict[str, AttributeValue] = {}
        attributes.update(self._get_agent_attributes())
        attributes.update(self._get_request_attributes())
        attributes.update(self._get_response_attributes())
        attributes.update(self._get_usage_attributes())
        attributes.update(self._get_content_attributes_for_span())
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._call_completion_hook(
            inputs=self.input_messages,
            outputs=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
        )
        self._record_metrics()

    @abstractmethod
    def _record_metrics(self) -> None:
        """Record invocation metrics."""


class LocalAgentInvocation(AgentInvocation):
    """Represents an in-process agent invocation (INTERNAL span kind).

    Use handler.invoke_local_agent() rather than constructing this directly.

    Reference:
        https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-internal-span
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        request_model: str | None = None,
        agent_name: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
        data: AgentData | None = None,
    ) -> None:
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            span_kind=SpanKind.INTERNAL,
            request_model=request_model,
            agent_name=agent_name,
            content_capturing_mode=content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
            data=data,
        )

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        if self.data.agent_name is not None:
            attrs[GenAI.GEN_AI_AGENT_NAME] = self.data.agent_name
        if self.data.request_model is not None:
            attrs[GenAI.GEN_AI_REQUEST_MODEL] = self.data.request_model
        attrs.update(self.metric_attributes)
        return attrs

    def _record_metrics(self) -> None:
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        self._instruments.invoke_agent_duration.record(
            duration_seconds,
            attributes=self._get_metric_attributes(),
            context=self._span_context,
        )


class RemoteAgentInvocation(AgentInvocation):
    """Represents a remote agent invocation (CLIENT span kind).

    Use handler.invoke_remote_agent() rather than constructing this directly.

    Reference:
        https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-client-span
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
        agent_name: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
        data: AgentData | None = None,
    ) -> None:
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                (GenAI.GEN_AI_AGENT_NAME, agent_name),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
            )
            if v is not None
        }
        if data is None:
            data = AgentData(
                agent_name=agent_name,
                provider_name=provider,
                request_model=request_model,
                server_address=server_address,
                server_port=server_port,
                conversation_id=conversation_id,
                input_messages=[],
                output_messages=[],
                system_instructions=[],
            )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            span_kind=SpanKind.CLIENT,
            request_model=request_model,
            agent_name=agent_name,
            start_attributes=start_attributes,
            content_capturing_mode=content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            conversation_id=conversation_id,
            data=data,
        )
        self._provider: str = provider
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port

    @property
    def provider(self) -> str:
        """The provider name provided at construction time."""
        return self.data.provider_name or self._provider

    @property
    def server_address(self) -> str | None:
        """The server address provided at construction time."""
        return self.data.server_address

    @property
    def server_port(self) -> int | None:
        """The server port provided at construction time."""
        return self.data.server_port

    @property
    def agent_id(self) -> str | None:
        return self.data.agent_id

    @agent_id.setter
    def agent_id(self, value: str | None) -> None:
        self.data.agent_id = value

    @property
    def agent_version(self) -> str | None:
        return self.data.agent_version

    @agent_version.setter
    def agent_version(self, value: str | None) -> None:
        self.data.agent_version = value

    @property
    def previous_response_id(self) -> str | None:
        return self.data.request_previous_response_id

    @previous_response_id.setter
    def previous_response_id(self, value: str | None) -> None:
        self.data.request_previous_response_id = value

    @property
    def cache_write_input_tokens(self) -> int | None:
        """The number of cache write input tokens."""
        return self.data.usage_cache_write_input_tokens

    @cache_write_input_tokens.setter
    def cache_write_input_tokens(self, value: int | None) -> None:
        self.data.usage_cache_write_input_tokens = value

    @property
    def cache_creation_input_tokens(self) -> int | None:
        """The number of cache creation input tokens.

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

    def _get_agent_attributes(self) -> dict[str, AttributeValue]:
        attrs = super()._get_agent_attributes()
        optional_attrs = (
            (GenAI.GEN_AI_AGENT_ID, self.data.agent_id),
            (GenAI.GEN_AI_AGENT_VERSION, self.data.agent_version),
        )
        attrs.update({k: v for k, v in optional_attrs if v is not None})
        return attrs

    def _get_request_attributes(self) -> dict[str, AttributeValue]:
        attrs = super()._get_request_attributes()
        optional_attrs = (
            (
                GenAI.GEN_AI_PROVIDER_NAME,
                self.data.provider_name or self._provider,
            ),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
        )
        attrs.update({k: v for k, v in optional_attrs if v is not None})
        if self.data.request_previous_response_id is not None:
            attrs[_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID] = (
                self.data.request_previous_response_id
            )
        return attrs

    def _get_usage_attributes(self) -> dict[str, AttributeValue]:
        attrs = super()._get_usage_attributes()
        if self.data.usage_cache_write_input_tokens is not None:
            attrs[_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] = (
                self.data.usage_cache_write_input_tokens
            )
        if self.data.usage_cache_read_input_tokens is not None:
            attrs[GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = (
                self.data.usage_cache_read_input_tokens
            )
        return attrs

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (
                GenAI.GEN_AI_PROVIDER_NAME,
                self.data.provider_name or self._provider,
            ),
            (
                GenAI.GEN_AI_REQUEST_MODEL,
                self.data.request_model or self._request_model,
            ),
            (
                server_attributes.SERVER_ADDRESS,
                self.data.server_address or self._server_address,
            ),
            (
                server_attributes.SERVER_PORT,
                self.data.server_port or self._server_port,
            ),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }
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

    def _record_metrics(self) -> None:
        self._record_client_metrics()
