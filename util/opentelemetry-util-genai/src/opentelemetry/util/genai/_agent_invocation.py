# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.semconv.gen_ai import (
    GenAiOperationName,
    GenAiTokenType,
)
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    InvokeAgentClientOperation,
    InvokeAgentInternalOperation,
)
from opentelemetry.util.genai.types import (
    MessagePart,
    SystemInstructionPart,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)


class LocalAgentInvocation(GenAIInvocation, InvokeAgentInternalOperation):
    """Represents an in-process agent invocation (INTERNAL span kind).

    Use handler.invoke_local_agent() rather than constructing this directly.

    Reference:
        https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-internal-span
    """

    temperature = _Attribute[float | None]("request_temperature")
    top_p = _Attribute[float | None]("request_top_p")
    frequency_penalty = _Attribute[float | None]("request_frequency_penalty")
    presence_penalty = _Attribute[float | None]("request_presence_penalty")
    max_tokens = _Attribute[int | None]("request_max_tokens")
    stop_sequences = _Attribute[list[str] | None]("request_stop_sequences")
    seed = _Attribute[int | None]("request_seed")
    choice_count = _Attribute[int | None]("request_choice_count")
    input_tokens = _Attribute[int | None]("usage_input_tokens")
    output_tokens = _Attribute[int | None]("usage_output_tokens")
    system_instruction = _Attribute[
        list[SystemInstructionPart] | Sequence[MessagePart]
    ]("system_instructions", default_factory=list)
    finish_reasons = _Attribute[list[str] | None]("response_finish_reasons")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        request_model: str | None = None,
        agent_name: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        operation_name = GenAiOperationName.INVOKE_AGENT.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        InvokeAgentInternalOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            completion_hook=completion_hook,
            operation_name=operation_name,
            request_model=request_model,
            agent_name=agent_name,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        if self.input_messages is None:
            self.input_messages = []
        if self.output_messages is None:
            self.output_messages = []
        self.start()


class RemoteAgentInvocation(GenAIInvocation, InvokeAgentClientOperation):
    """Represents a remote agent invocation (CLIENT span kind).

    Use handler.invoke_remote_agent() rather than constructing this directly.

    Reference:
        https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md#invoke-agent-client-span
    """

    _provider = _Attribute[str]("provider_name")
    previous_response_id = _Attribute[str | None](
        "request_previous_response_id"
    )
    cache_read_input_tokens = _Attribute[int | None](
        "usage_cache_read_input_tokens"
    )
    cache_write_input_tokens = _Attribute[int | None](
        "usage_cache_write_input_tokens"
    )
    temperature = _Attribute[float | None]("request_temperature")
    top_p = _Attribute[float | None]("request_top_p")
    frequency_penalty = _Attribute[float | None]("request_frequency_penalty")
    presence_penalty = _Attribute[float | None]("request_presence_penalty")
    max_tokens = _Attribute[int | None]("request_max_tokens")
    stop_sequences = _Attribute[list[str] | None]("request_stop_sequences")
    seed = _Attribute[int | None]("request_seed")
    choice_count = _Attribute[int | None]("request_choice_count")
    input_tokens = _Attribute[int | None]("usage_input_tokens")
    output_tokens = _Attribute[int | None]("usage_output_tokens")
    system_instruction = _Attribute[
        list[SystemInstructionPart] | Sequence[MessagePart]
    ]("system_instructions", default_factory=list)
    finish_reasons = _Attribute[list[str] | None]("response_finish_reasons")

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
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        operation_name = GenAiOperationName.INVOKE_AGENT.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        InvokeAgentClientOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            completion_hook=completion_hook,
            operation_name=operation_name,
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            agent_name=agent_name,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        if self.input_messages is None:
            self.input_messages = []
        if self.output_messages is None:
            self.output_messages = []
        self.start()

    @property
    def cache_creation_input_tokens(self) -> int | None:
        """The number of cache creation input tokens.

        .. deprecated:: 1.3b0
            Use :attr:`cache_write_input_tokens` instead.
        """
        return self.cache_write_input_tokens

    @cache_creation_input_tokens.setter
    def cache_creation_input_tokens(self, value: int | None) -> None:
        self.cache_write_input_tokens = value

    def _on_stream_chunk(self, chunk_at: float) -> None:
        pass

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


class AgentInvocation(GenAIInvocation, ABC):
    """Base class representing a GenAI agent invocation (invoke_agent span).

    .. deprecated:: 1.4b0
        Use :class:`LocalAgentInvocation` or :class:`RemoteAgentInvocation` instead.
    """


AgentInvocation.register(LocalAgentInvocation)
AgentInvocation.register(RemoteAgentInvocation)
