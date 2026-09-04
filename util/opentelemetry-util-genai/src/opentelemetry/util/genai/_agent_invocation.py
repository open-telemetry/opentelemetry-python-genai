# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from typing import Final

from opentelemetry._logs import Logger
from opentelemetry.metrics import Meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._invocation import (
    Error,
    GenAIInvocation,
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
from opentelemetry.util.types import AttributeValue

_GEN_AI_INVOKE_AGENT_DURATION: Final = "gen_ai.invoke_agent.duration"
_GEN_AI_INVOKE_AGENT_DURATION_BUCKETS: Final = [
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    3.2,
    6.4,
    12.8,
    25.6,
    51.2,
    102.4,
    204.8,
    409.6,
]

_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS: Final = (
    "gen_ai.usage.cache_write.input_tokens"
)
_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID: Final = (
    "gen_ai.request.previous_response.id"
)


class AgentInvocation(GenAIInvocation):
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
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        span_kind: SpanKind,
        request_model: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        _operation_name = GenAI.GenAiOperationNameValues.INVOKE_AGENT.value
        super().__init__(
            tracer,
            meter,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {agent_name}"
            if agent_name
            else _operation_name,
            span_kind=span_kind,
        )
        self._request_model: str | None = request_model
        self._agent_name: str | None = agent_name
        self.agent_description: str | None = None

        self.conversation_id: str | None = None
        self.data_source_id: str | None = None
        self.output_type: str | None = None

        self.temperature: float | None = None
        self.top_p: float | None = None
        self.frequency_penalty: float | None = None
        self.presence_penalty: float | None = None
        self.max_tokens: int | None = None
        self.stop_sequences: list[str] | None = None
        self.seed: int | None = None
        self.choice_count: int | None = None

        self.finish_reasons: list[str] | None = None

        self.input_tokens: int | None = None
        self.output_tokens: int | None = None

        self.input_messages: list[InputMessage] = []
        self.output_messages: list[OutputMessage] = []
        self.system_instruction: (
            list[SystemInstructionPart] | list[MessagePart]
        ) = []
        """System instructions for the agent. Passing ``MessagePart`` is deprecated; use ``SystemInstructionPart``."""
        self.tool_definitions: list[ToolDefinition] | None = None

    @property
    def agent_name(self) -> str | None:
        """The agent name provided at construction time."""
        return self._agent_name

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
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, self.input_tokens),
            (GenAI.GEN_AI_USAGE_OUTPUT_TOKENS, self.output_tokens),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _get_content_attributes_for_span(self) -> dict[str, AttributeValue]:
        return get_content_attributes(
            input_messages=self.input_messages,
            output_messages=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
            for_span=True,
        )


class LocalAgentInvocation(AgentInvocation):
    """Represents an in-process agent invocation (INTERNAL span kind).

    Use handler.invoke_local_agent() rather than constructing this directly.

    Reference:
        https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-internal-span
    """

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        request_model: str | None = None,
        agent_name: str | None = None,
    ) -> None:
        super().__init__(
            tracer,
            meter,
            logger,
            completion_hook,
            span_kind=SpanKind.INTERNAL,
            request_model=request_model,
            agent_name=agent_name,
        )
        self._start(self._get_start_attributes())

    def _get_start_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_REQUEST_MODEL, self._request_model),
            (GenAI.GEN_AI_AGENT_NAME, self._agent_name),
        )
        return {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        attrs: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            attrs[GenAI.GEN_AI_AGENT_NAME] = self._agent_name
        if self._request_model is not None:
            attrs[GenAI.GEN_AI_REQUEST_MODEL] = self._request_model
        attrs.update(self.metric_attributes)
        return attrs

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)

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

    def _record_metrics(self) -> None:
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        histogram = self._meter.create_histogram(
            name=_GEN_AI_INVOKE_AGENT_DURATION,
            description="Measures the duration of an in-process agent invocation.",
            unit="s",
            explicit_bucket_boundaries_advisory=_GEN_AI_INVOKE_AGENT_DURATION_BUCKETS,
        )
        histogram.record(
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
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
        agent_version: str | None = None,
    ) -> None:
        super().__init__(
            tracer,
            meter,
            logger,
            completion_hook,
            span_kind=SpanKind.CLIENT,
            request_model=request_model,
            agent_name=agent_name,
        )
        self._provider: str = provider
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port

        self.agent_id: str | None = agent_id
        self.agent_version: str | None = agent_version
        self.previous_response_id: str | None = None
        self._cache_write_input_tokens: int | None = None
        self.cache_read_input_tokens: int | None = None

        self._start(self._get_start_attributes())

    @property
    def cache_write_input_tokens(self) -> int | None:
        """The number of cache write input tokens."""
        return self._cache_write_input_tokens

    @cache_write_input_tokens.setter
    def cache_write_input_tokens(self, value: int | None) -> None:
        self._cache_write_input_tokens = value

    @property
    def cache_creation_input_tokens(self) -> int | None:
        """The number of cache creation input tokens.

        .. deprecated:: 1.3b0
            Use :attr:`cache_write_input_tokens` instead.
        """
        return self._cache_write_input_tokens

    @cache_creation_input_tokens.setter
    def cache_creation_input_tokens(self, value: int | None) -> None:
        self._cache_write_input_tokens = value

    def _get_start_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_REQUEST_MODEL, self._request_model),
            (GenAI.GEN_AI_AGENT_NAME, self._agent_name),
            (server_attributes.SERVER_ADDRESS, self._server_address),
            (server_attributes.SERVER_PORT, self._server_port),
            (GenAI.GEN_AI_PROVIDER_NAME, self._provider),
        )
        return {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }

    def _get_agent_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_AGENT_ID, self.agent_id),
            (GenAI.GEN_AI_AGENT_DESCRIPTION, self.agent_description),
            (GenAI.GEN_AI_AGENT_VERSION, self.agent_version),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _get_request_attributes(self) -> dict[str, AttributeValue]:
        attrs = dict(super()._get_request_attributes())
        if self.previous_response_id is not None:
            attrs[_GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID] = (
                self.previous_response_id
            )
        return attrs

    def _get_usage_attributes(self) -> dict[str, AttributeValue]:
        attrs = dict(super()._get_usage_attributes())
        if self.cache_write_input_tokens is not None:
            attrs[_GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] = (
                self.cache_write_input_tokens
            )
        if self.cache_read_input_tokens is not None:
            attrs[GenAI.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = (
                self.cache_read_input_tokens
            )
        return attrs

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_PROVIDER_NAME, self._provider),
            (GenAI.GEN_AI_REQUEST_MODEL, self._request_model),
            (server_attributes.SERVER_ADDRESS, self._server_address),
            (server_attributes.SERVER_PORT, self._server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _get_metric_token_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.input_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.INPUT.value] = self.input_tokens
        if self.output_tokens is not None:
            counts[GenAI.GenAiTokenTypeValues.OUTPUT.value] = (
                self.output_tokens
            )
        return counts

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)

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
        self._record_client_metrics()
