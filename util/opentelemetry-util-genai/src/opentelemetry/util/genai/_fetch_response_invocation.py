# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from opentelemetry._logs import Logger
from opentelemetry.context import Context, get_value
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
    _resolve_suppressed_content_and_hook,
    get_content_attributes,
)
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    ErrorTypeResolver,
    MessagePart,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import ContentCapturingMode
from opentelemetry.util.types import AttributeValue

CLIENT_FETCH_RESPONSE_CONTEXT_KEY: Final[str] = (
    "opentelemetry.genai.client.fetch_response.context"
)

# TODO: Migrate to gen_ai_attributes constants once available in the semconv
# package. Added to the GenAI semantic conventions in
# https://github.com/open-telemetry/semantic-conventions-genai/pull/353.
_FETCH_RESPONSE_OPERATION_NAME: Final = "fetch_response"
_GEN_AI_REQUEST_STREAM_CURSOR: Final = "gen_ai.request.stream_cursor"
_GEN_AI_RESPONSE_STATUS: Final = "gen_ai.response.status"


@dataclass
class FetchResponseData(_ContextData):
    """Typed data passed from inner fetch response invocations to the outer invocation."""

    provider_name: str | None = None
    response_id: str | None = None
    request_stream: bool | None = None
    server_address: str | None = None
    server_port: int | None = None
    response_model: str | None = None
    response_status: str | None = None
    response_finish_reasons: list[str] | None = None
    request_stream_cursor: str | None = None
    output_messages: Sequence[OutputMessage] | None = None
    system_instructions: (
        Sequence[SystemInstructionPart] | Sequence[MessagePart] | None
    ) = None
    tool_definitions: list[ToolDefinition] | None = None


class FetchResponseInvocation(GenAIInvocation):
    """Represents a single fetch of a previously generated model response.

    Use handler.fetch_response() rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#fetch-response

    The operation performs no inference and consumes no tokens: it returns a
    response produced by an earlier operation. Any token counts carried on the
    fetched response describe that original generation and MUST NOT be reported
    here, so this invocation deliberately exposes no token usage fields.

    Semantic convention attributes for fetch response spans:
    - gen_ai.operation.name: "fetch_response" (Required)
    - gen_ai.provider.name: Provider name (Required)
    - gen_ai.response.id: Identifier of the response being fetched (Required)
    - error.type: Error type when the fetch itself failed (Conditionally Required)
    - gen_ai.request.stream_cursor: Set from ``stream_cursor`` when the fetch
      resumes a streamed response from a prior position (Conditionally Required)
    - gen_ai.request.stream: Set to true when the fetched response is streamed
      (Conditionally Required)
    - server.port: Set only when ``server_port`` is provided (Conditionally Required)
    - gen_ai.response.finish_reasons: Outcome of the original generation
      (Recommended)
    - gen_ai.response.model: Set from ``response_model_name`` (Recommended)
    - gen_ai.response.status: Lifecycle status of the fetched response
      (Recommended)
    - server.address: Set only when ``server_address`` is provided (Recommended)
    - gen_ai.output.messages, gen_ai.system_instructions,
      gen_ai.tool.definitions: content carried on the fetched response,
      recorded on the span only when content capturing is enabled (Opt-In).
      A fetched response does not carry the original input messages, so
      gen_ai.input.messages is never set.

    A fetched response whose *original* generation failed is not a failure of
    the fetch: report it through ``response_status`` and ``finish_reasons`` and
    still call ``stop()``. Only call ``fail()`` when the fetch call itself
    failed.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        response_id: str,
        request_stream: bool | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        start_span: bool = True,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        """Use handler.fetch_response() rather than calling this directly."""
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (GenAI.GEN_AI_RESPONSE_ID, response_id),
                (GenAI.GEN_AI_REQUEST_STREAM, request_stream),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
            )
            if v is not None
        }
        self._data: FetchResponseData = FetchResponseData(
            provider_name=provider,
            response_id=response_id,
            request_stream=request_stream,
            server_address=server_address,
            server_port=server_port,
            output_messages=[],
            system_instructions=[],
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_FETCH_RESPONSE_OPERATION_NAME,
            # The response identifier is high cardinality, so semconv keeps it
            # out of the span name.
            span_name=_FETCH_RESPONSE_OPERATION_NAME,
            span_kind=SpanKind.CLIENT,
            error_type_resolver=error_type_resolver,
            start_attributes=start_attributes,
            content_capturing_mode=content_capturing_mode,
            start_span=start_span,
            context=context,
            _attach_to_context=_attach_to_context,
            attributes=self._data.attributes,
            metric_attributes=self._data.metric_attributes,
            context_key=CLIENT_FETCH_RESPONSE_CONTEXT_KEY,
            dataclass_class_object=FetchResponseData,
        )
        self._provider: str = provider
        self._response_id: str = response_id
        self._request_stream = request_stream
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self._data.attributes = self.attributes
        self._data.metric_attributes = self.metric_attributes

    @property
    def response_id(self) -> str:
        """The identifier of the response being fetched."""
        return self._data.response_id or self._response_id

    @property
    def provider(self) -> str | None:
        return self._data.provider_name

    @property
    def server_address(self) -> str | None:
        return self._data.server_address

    @property
    def server_port(self) -> int | None:
        return self._data.server_port

    @property
    def request_stream(self) -> bool | None:
        return (
            self._request_stream
            if self._request_stream is not None
            else self._data.request_stream
        )

    @request_stream.setter
    def request_stream(self, value: bool | None) -> None:
        self._request_stream = value
        self._data.request_stream = value

    @property
    def response_model_name(self) -> str | None:
        return self._data.response_model

    @response_model_name.setter
    def response_model_name(self, value: str | None) -> None:
        self._data.response_model = value

    @property
    def response_status(self) -> str | None:
        return self._data.response_status

    @response_status.setter
    def response_status(self, value: str | None) -> None:
        self._data.response_status = value

    @property
    def finish_reasons(self) -> list[str] | None:
        return self._data.response_finish_reasons

    @finish_reasons.setter
    def finish_reasons(self, value: Sequence[str] | None) -> None:
        self._data.response_finish_reasons = (
            list(value) if value is not None else None
        )

    @property
    def stream_cursor(self) -> str | None:
        return self._data.request_stream_cursor

    @stream_cursor.setter
    def stream_cursor(self, value: str | None) -> None:
        self._data.request_stream_cursor = value

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

    def enrich_from_context(self, data: FetchResponseData) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values. Inner
        invocations never override content capture fields unless explicitly enabled.
        """
        output_messages = self._data.output_messages
        system_instructions = self._data.system_instructions
        tool_definitions = self._data.tool_definitions

        self._data.merge(data, overwrite=False)

        if data.capture_content:
            self._data.output_messages = (
                output_messages if output_messages else data.output_messages
            )
            self._data.system_instructions = (
                system_instructions
                if system_instructions
                else data.system_instructions
            )
            self._data.tool_definitions = (
                tool_definitions if tool_definitions else data.tool_definitions
            )
        else:
            self._data.output_messages = output_messages
            self._data.system_instructions = system_instructions
            self._data.tool_definitions = tool_definitions

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        # response_id intentionally excluded — high cardinality.
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_RESPONSE_MODEL, self._data.response_model),
            (server_attributes.SERVER_ADDRESS, self._data.server_address),
            (server_attributes.SERVER_PORT, self._data.server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            GenAI.GEN_AI_PROVIDER_NAME: self._data.provider_name
            or self._provider,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _get_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (
                GenAI.GEN_AI_PROVIDER_NAME,
                self._data.provider_name or self._provider,
            ),
            (server_attributes.SERVER_ADDRESS, self._data.server_address),
            (server_attributes.SERVER_PORT, self._data.server_port),
            (_GEN_AI_REQUEST_STREAM_CURSOR, self._data.request_stream_cursor),
            (
                GenAI.GEN_AI_RESPONSE_FINISH_REASONS,
                self._data.response_finish_reasons or None,
            ),
            (GenAI.GEN_AI_RESPONSE_MODEL, self._data.response_model),
            (_GEN_AI_RESPONSE_STATUS, self._data.response_status),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_value(
            CLIENT_FETCH_RESPONSE_CONTEXT_KEY, context=self._span_context
        )
        if isinstance(ctx_data, FetchResponseData):
            self.enrich_from_context(ctx_data)
        self._data.attributes = self.attributes
        self._data.metric_attributes = self.metric_attributes
        attributes = self._get_attributes()
        attributes.update(
            get_content_attributes(
                # A fetched response does not carry the original request's
                # input messages.
                input_messages=(),
                output_messages=self.output_messages,
                system_instruction=self.system_instruction,
                tool_definitions=self.tool_definitions,
                for_span=True,
                content_capturing_mode=self._content_capturing_mode,
            )
        )
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._record_client_metrics()
        self._call_completion_hook(
            outputs=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
        )


class SuppressedFetchResponseInvocation(FetchResponseInvocation):
    """Represents a fetch response invocation running inside an active fetch response context.

    Suppresses span creation and metrics. On stop or fail, publishes its
    attributes to the active fetch response context.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        response_id: str,
        request_stream: bool | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        ctx_data = get_value(
            CLIENT_FETCH_RESPONSE_CONTEXT_KEY, context=context
        )
        (
            self._capture_content_to_context,
            self._upload_completion,
            effective_content_mode,
            effective_completion_hook,
        ) = _resolve_suppressed_content_and_hook(
            ctx_data, content_capturing_mode, completion_hook
        )

        super().__init__(
            tracer,
            instruments,
            logger,
            effective_completion_hook,
            provider,
            response_id=response_id,
            request_stream=request_stream,
            server_address=server_address,
            server_port=server_port,
            error_type_resolver=error_type_resolver,
            content_capturing_mode=effective_content_mode,
            start_span=False,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def _on_stream_chunk(self, chunk_at: float) -> None:
        super()._on_stream_chunk(chunk_at)
        self._data.request_stream = True

    def publish_to_context(self, data: FetchResponseData) -> None:
        """Publish invocation attributes to the active fetch response context."""
        self._data.attributes = self.attributes
        self._data.metric_attributes = self.metric_attributes
        if self._request_stream is not None:
            self._data.request_stream = self._request_stream

        # Prevent inner content from merging into context when content capture is off,
        # such as when messages were populated only for completion hook upload.
        if not self._capture_content_to_context:
            self._data.output_messages = None
            self._data.system_instructions = None
            self._data.tool_definitions = None

        data.merge(self._data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True

        if self._upload_completion:
            self._call_completion_hook(
                outputs=self.output_messages,
                system_instruction=self.system_instruction,
                tool_definitions=self.tool_definitions,
                log_record=None,
            )

        ctx_data = get_value(
            CLIENT_FETCH_RESPONSE_CONTEXT_KEY, context=self._span_context
        )
        if isinstance(ctx_data, FetchResponseData):
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass
