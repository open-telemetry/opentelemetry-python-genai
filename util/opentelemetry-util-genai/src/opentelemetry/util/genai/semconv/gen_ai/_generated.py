# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
# Code generated from OpenTelemetry GenAI semantic conventions. DO NOT EDIT.

from __future__ import annotations

import timeit
from collections import ChainMap
from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import cached_property
from types import TracebackType
from typing import cast

from typing_extensions import Self

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.trace import (
    INVALID_SPAN,
    Span,
    SpanKind,
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
    use_span,
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.semconv.error import attributes as ErrorAttr
from opentelemetry.util.genai.semconv.gen_ai import attributes as Attr
from opentelemetry.util.genai.semconv.server import attributes as ServerAttr
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    Error,
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    gen_ai_json_dumps,
)
from opentelemetry.util.types import AnyValue, AttributeValue


def _value(value: AttributeValue | Enum) -> AttributeValue:
    return value.value if isinstance(value, Enum) else value


def _span_name_value(value: object | None) -> str | None:
    if value is None:
        return None
    v = value.value if isinstance(value, Enum) else value
    if not v or v == "_OTHER":
        return None
    return str(v)


def _set_span_json_attribute(
    span: Span, name: str, value: object | None
) -> None:
    if value is None:
        return
    if isinstance(value, (bool, str, bytes, int, float)):
        span.set_attribute(name, _value(value))
        return
    span.set_attribute(name, gen_ai_json_dumps(value))


def _combine_attributes(
    typed_attributes: dict[str, AttributeValue],
    additional_attributes: Mapping[str, AttributeValue] | None,
) -> Mapping[str, AttributeValue]:
    if not additional_attributes:
        return typed_attributes
    if not typed_attributes:
        return additional_attributes
    return ChainMap(
        typed_attributes,
        cast("MutableMapping[str, AttributeValue]", additional_attributes),
    )


def _structured_value(value: object) -> AnyValue:
    if is_dataclass(value) and not isinstance(value, type):
        return _structured_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        mapping = cast("Mapping[str, object]", value)
        return {
            name: _structured_value(item) for name, item in mapping.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        sequence = cast("Sequence[object]", value)
        return [_structured_value(item) for item in sequence]
    return cast("AnyValue", value)


class EmbeddingsClientOperation:
    """`gen_ai.embeddings.client` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        error_type_resolver: ErrorTypeResolver | None = None,
        embeddings_dimension_count: int | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        provider_name: Attr.GenAIProviderName | str | None = None,
        request_encoding_formats: Sequence[str] | None = None,
        request_model: str | None = None,
        response_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        usage_input_tokens: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.embeddings_dimension_count: int | None = (
            embeddings_dimension_count
        )
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self._provider_name: Attr.GenAIProviderName | str | None = (
            provider_name
        )
        self.request_encoding_formats: Sequence[str] | None = (
            request_encoding_formats
        )
        self._request_model: str | None = request_model
        self.response_model: str | None = response_model
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.usage_input_tokens: int | None = usage_input_tokens
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        operation_name = _span_name_value(self._operation_name)
        request_model = _span_name_value(self._request_model)
        if operation_name and request_model:
            return f"{operation_name} {request_model}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.embeddings.client"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._provider_name is not None:
            sampling_attributes[Attr.GEN_AI_PROVIDER_NAME] = _value(
                self._provider_name
            )
        if self._request_model is not None:
            sampling_attributes[Attr.GEN_AI_REQUEST_MODEL] = _value(
                self._request_model
            )
        if self._server_address is not None:
            sampling_attributes[ServerAttr.SERVER_ADDRESS] = _value(
                self._server_address
            )
        if self._server_port is not None:
            sampling_attributes[ServerAttr.SERVER_PORT] = _value(
                self._server_port
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.embeddings_dimension_count is not None:
            span.set_attribute(
                Attr.GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
                _value(self.embeddings_dimension_count),
            )
        if self.request_encoding_formats is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_ENCODING_FORMATS,
                _value(self.request_encoding_formats),
            )
        if self.response_model is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_MODEL,
                _value(self.response_model),
            )
        if self.usage_input_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_INPUT_TOKENS,
                _value(self.usage_input_tokens),
            )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_client_operation_duration(duration_s, context=ctx)
        span.end(end_time=end_time)

    def _record_client_operation_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_token_usage(
        self,
        value: int,
        token_type: Attr.GenAITokenType | str,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {
            Attr.GEN_AI_TOKEN_TYPE: _value(token_type),
        }
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_token_usage_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _client_operation_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_token_usage_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used.",
            explicit_bucket_boundaries_advisory=[
                1,
                4,
                16,
                64,
                256,
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                67108864,
            ],
        )


class ExecuteToolInternalOperation:
    """`gen_ai.execute_tool.internal` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        error_type_resolver: ErrorTypeResolver | None = None,
        agent_name: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        tool_call_arguments: AnyValue | None = None,
        tool_call_id: str | None = None,
        tool_call_result: AnyValue | None = None,
        tool_description: str | None = None,
        tool_name: str | None = None,
        tool_type: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self._agent_name: str | None = agent_name
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.tool_call_arguments: AnyValue | None = tool_call_arguments
        self.tool_call_id: str | None = tool_call_id
        self.tool_call_result: AnyValue | None = tool_call_result
        self.tool_description: str | None = tool_description
        self._tool_name: str | None = tool_name
        self._tool_type: str | None = tool_type
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        operation_name = _span_name_value(self._operation_name)
        tool_name = _span_name_value(self._tool_name)
        if operation_name and tool_name:
            return f"{operation_name} {tool_name}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.execute_tool.internal"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            sampling_attributes[Attr.GEN_AI_AGENT_NAME] = _value(
                self._agent_name
            )
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._tool_name is not None:
            sampling_attributes[Attr.GEN_AI_TOOL_NAME] = _value(
                self._tool_name
            )
        if self._tool_type is not None:
            sampling_attributes[Attr.GEN_AI_TOOL_TYPE] = _value(
                self._tool_type
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.INTERNAL,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.tool_call_id is not None:
            span.set_attribute(
                Attr.GEN_AI_TOOL_CALL_ID,
                _value(self.tool_call_id),
            )
        if self.tool_description is not None:
            span.set_attribute(
                Attr.GEN_AI_TOOL_DESCRIPTION,
                _value(self.tool_description),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.tool_call_arguments is not None:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_CALL_ARGUMENTS,
                    self.tool_call_arguments,
                )
            if self.tool_call_result is not None:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_CALL_RESULT,
                    self.tool_call_result,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_execute_tool_duration(duration_s, context=ctx)
        span.end(end_time=end_time)

    def _record_execute_tool_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._agent_name is not None:
            attrs[Attr.GEN_AI_AGENT_NAME] = _value(self._agent_name)
        if self._tool_name is not None:
            attrs[Attr.GEN_AI_TOOL_NAME] = _value(self._tool_name)
        if self._tool_type is not None:
            attrs[Attr.GEN_AI_TOOL_TYPE] = _value(self._tool_type)
        self._execute_tool_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _execute_tool_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.execute_tool.duration",
            unit="s",
            description="The duration of a single tool execution.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )


class FetchResponseClientOperation:
    """`gen_ai.fetch_response.client` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        completion_hook: CompletionHook | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        output_messages: Sequence[OutputMessage] | None = None,
        provider_name: Attr.GenAIProviderName | str | None = None,
        request_stream_cursor: str | None = None,
        response_finish_reasons: Sequence[str] | None = None,
        response_id: str | None = None,
        response_model: str | None = None,
        response_status: Attr.GenAIResponseStatus | str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        system_instructions: Sequence[SystemInstructionPart] | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self.completion_hook: CompletionHook | None = completion_hook
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.output_messages: Sequence[OutputMessage] | None = output_messages
        self._provider_name: Attr.GenAIProviderName | str | None = (
            provider_name
        )
        self.request_stream_cursor: str | None = request_stream_cursor
        self.response_finish_reasons: Sequence[str] | None = (
            response_finish_reasons
        )
        self.response_id: str | None = response_id
        self.response_model: str | None = response_model
        self.response_status: Attr.GenAIResponseStatus | str | None = (
            response_status
        )
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.system_instructions: Sequence[SystemInstructionPart] | None = (
            system_instructions
        )
        self.tool_definitions: Sequence[ToolDefinition] | None = (
            tool_definitions
        )
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or (
            self.completion_hook is not None
            and not isinstance(self.completion_hook, _NoOpCompletionHook)
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        operation_name = _span_name_value(self._operation_name)
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.fetch_response.client"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._provider_name is not None:
            sampling_attributes[Attr.GEN_AI_PROVIDER_NAME] = _value(
                self._provider_name
            )
        if self._server_address is not None:
            sampling_attributes[ServerAttr.SERVER_ADDRESS] = _value(
                self._server_address
            )
        if self._server_port is not None:
            sampling_attributes[ServerAttr.SERVER_PORT] = _value(
                self._server_port
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.request_stream_cursor is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_STREAM_CURSOR,
                _value(self.request_stream_cursor),
            )
        if self.response_finish_reasons is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_FINISH_REASONS,
                _value(self.response_finish_reasons),
            )
        if self.response_id is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_ID,
                _value(self.response_id),
            )
        if self.response_model is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_MODEL,
                _value(self.response_model),
            )
        if self.response_status is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_STATUS,
                _value(self.response_status),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.output_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_OUTPUT_MESSAGES,
                    self.output_messages,
                )
            if self.system_instructions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_SYSTEM_INSTRUCTIONS,
                    self.system_instructions,
                )
            if self.tool_definitions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_DEFINITIONS,
                    self.tool_definitions,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_client_operation_duration(duration_s, context=ctx)
        if self.completion_hook is not None:
            self.completion_hook.on_completion(
                inputs=[],
                outputs=(
                    list(self.output_messages) if self.output_messages else []
                ),
                system_instruction=cast(
                    "list[MessagePart]",
                    list(self.system_instructions)
                    if self.system_instructions
                    else [],
                ),
                tool_definitions=(
                    list(self.tool_definitions)
                    if self.tool_definitions
                    else None
                ),
                span=span,
                log_record=None,
            )
        span.end(end_time=end_time)

    def _record_client_operation_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_time_per_output_chunk(
        self,
        value: float,
        *,
        request_model: str | None = None,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(request_model)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_time_per_output_chunk_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_time_to_first_chunk(
        self,
        value: float,
        *,
        request_model: str | None = None,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(request_model)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_time_to_first_chunk_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _client_operation_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_operation_time_per_output_chunk_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.time_per_output_chunk",
            unit="s",
            description="Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_operation_time_to_first_chunk_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            description="Time to receive the first chunk, measured from when the client issues the generation request to when the first chunk is received in the response stream.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )


class InferenceClientOperation:
    """`gen_ai.inference.client` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        completion_hook: CompletionHook | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        conversation_compacted: bool | None = None,
        conversation_id: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        input_messages: Sequence[InputMessage] | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        output_messages: Sequence[OutputMessage] | None = None,
        output_type: Attr.GenAIOutputType | str | None = None,
        prompt_name: str | None = None,
        prompt_variable: Mapping[str, object] | None = None,
        prompt_version: str | None = None,
        provider_name: Attr.GenAIProviderName | str | None = None,
        request_choice_count: int | None = None,
        request_frequency_penalty: float | None = None,
        request_max_tokens: int | None = None,
        request_model: str | None = None,
        request_presence_penalty: float | None = None,
        request_previous_response_id: str | None = None,
        request_reasoning_level: str | None = None,
        request_seed: int | None = None,
        request_stop_sequences: Sequence[str] | None = None,
        request_stream: bool | None = None,
        request_temperature: float | None = None,
        request_top_k: int | None = None,
        request_top_p: float | None = None,
        response_finish_reasons: Sequence[str] | None = None,
        response_id: str | None = None,
        response_model: str | None = None,
        response_time_to_first_chunk: float | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        system_instructions: Sequence[SystemInstructionPart] | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        usage_audio_cache_read_input_tokens: int | None = None,
        usage_audio_input_tokens: int | None = None,
        usage_audio_output_tokens: int | None = None,
        usage_cache_read_input_tokens: int | None = None,
        usage_cache_write_input_tokens: int | None = None,
        usage_image_cache_read_input_tokens: int | None = None,
        usage_image_input_tokens: int | None = None,
        usage_image_output_tokens: int | None = None,
        usage_input_tokens: int | None = None,
        usage_output_tokens: int | None = None,
        usage_reasoning_output_tokens: int | None = None,
        usage_text_cache_read_input_tokens: int | None = None,
        usage_text_input_tokens: int | None = None,
        usage_text_output_tokens: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self.completion_hook: CompletionHook | None = completion_hook
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.conversation_compacted: bool | None = conversation_compacted
        self.conversation_id: str | None = conversation_id
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self.input_messages: Sequence[InputMessage] | None = input_messages
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.output_messages: Sequence[OutputMessage] | None = output_messages
        self.output_type: Attr.GenAIOutputType | str | None = output_type
        self.prompt_name: str | None = prompt_name
        self.prompt_variable: Mapping[str, object] | None = prompt_variable
        self.prompt_version: str | None = prompt_version
        self._provider_name: Attr.GenAIProviderName | str | None = (
            provider_name
        )
        self.request_choice_count: int | None = request_choice_count
        self.request_frequency_penalty: float | None = (
            request_frequency_penalty
        )
        self.request_max_tokens: int | None = request_max_tokens
        self._request_model: str | None = request_model
        self.request_presence_penalty: float | None = request_presence_penalty
        self.request_previous_response_id: str | None = (
            request_previous_response_id
        )
        self.request_reasoning_level: str | None = request_reasoning_level
        self.request_seed: int | None = request_seed
        self.request_stop_sequences: Sequence[str] | None = (
            request_stop_sequences
        )
        self.request_stream: bool | None = request_stream
        self.request_temperature: float | None = request_temperature
        self.request_top_k: int | None = request_top_k
        self.request_top_p: float | None = request_top_p
        self.response_finish_reasons: Sequence[str] | None = (
            response_finish_reasons
        )
        self.response_id: str | None = response_id
        self.response_model: str | None = response_model
        self.response_time_to_first_chunk: float | None = (
            response_time_to_first_chunk
        )
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.system_instructions: Sequence[SystemInstructionPart] | None = (
            system_instructions
        )
        self.tool_definitions: Sequence[ToolDefinition] | None = (
            tool_definitions
        )
        self.usage_audio_cache_read_input_tokens: int | None = (
            usage_audio_cache_read_input_tokens
        )
        self.usage_audio_input_tokens: int | None = usage_audio_input_tokens
        self.usage_audio_output_tokens: int | None = usage_audio_output_tokens
        self.usage_cache_read_input_tokens: int | None = (
            usage_cache_read_input_tokens
        )
        self.usage_cache_write_input_tokens: int | None = (
            usage_cache_write_input_tokens
        )
        self.usage_image_cache_read_input_tokens: int | None = (
            usage_image_cache_read_input_tokens
        )
        self.usage_image_input_tokens: int | None = usage_image_input_tokens
        self.usage_image_output_tokens: int | None = usage_image_output_tokens
        self.usage_input_tokens: int | None = usage_input_tokens
        self.usage_output_tokens: int | None = usage_output_tokens
        self.usage_reasoning_output_tokens: int | None = (
            usage_reasoning_output_tokens
        )
        self.usage_text_cache_read_input_tokens: int | None = (
            usage_text_cache_read_input_tokens
        )
        self.usage_text_input_tokens: int | None = usage_text_input_tokens
        self.usage_text_output_tokens: int | None = usage_text_output_tokens
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or (
            self.completion_hook is not None
            and not isinstance(self.completion_hook, _NoOpCompletionHook)
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        operation_name = _span_name_value(self._operation_name)
        request_model = _span_name_value(self._request_model)
        if operation_name and request_model:
            return f"{operation_name} {request_model}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.inference.client"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._provider_name is not None:
            sampling_attributes[Attr.GEN_AI_PROVIDER_NAME] = _value(
                self._provider_name
            )
        if self._request_model is not None:
            sampling_attributes[Attr.GEN_AI_REQUEST_MODEL] = _value(
                self._request_model
            )
        if self._server_address is not None:
            sampling_attributes[ServerAttr.SERVER_ADDRESS] = _value(
                self._server_address
            )
        if self._server_port is not None:
            sampling_attributes[ServerAttr.SERVER_PORT] = _value(
                self._server_port
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.conversation_compacted:
            span.set_attribute(
                Attr.GEN_AI_CONVERSATION_COMPACTED,
                _value(self.conversation_compacted),
            )
        if self.conversation_id is not None:
            span.set_attribute(
                Attr.GEN_AI_CONVERSATION_ID,
                _value(self.conversation_id),
            )
        if self.output_type is not None:
            span.set_attribute(
                Attr.GEN_AI_OUTPUT_TYPE,
                _value(self.output_type),
            )
        if self.prompt_name is not None:
            span.set_attribute(
                Attr.GEN_AI_PROMPT_NAME,
                _value(self.prompt_name),
            )
        if self.prompt_version is not None:
            span.set_attribute(
                Attr.GEN_AI_PROMPT_VERSION,
                _value(self.prompt_version),
            )
        if self.request_choice_count is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_CHOICE_COUNT,
                _value(self.request_choice_count),
            )
        if self.request_frequency_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                _value(self.request_frequency_penalty),
            )
        if self.request_max_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_MAX_TOKENS,
                _value(self.request_max_tokens),
            )
        if self.request_presence_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_PRESENCE_PENALTY,
                _value(self.request_presence_penalty),
            )
        if self.request_previous_response_id is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
                _value(self.request_previous_response_id),
            )
        if self.request_reasoning_level is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_REASONING_LEVEL,
                _value(self.request_reasoning_level),
            )
        if self.request_seed is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_SEED,
                _value(self.request_seed),
            )
        if self.request_stop_sequences is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_STOP_SEQUENCES,
                _value(self.request_stop_sequences),
            )
        if self.request_stream is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_STREAM,
                _value(self.request_stream),
            )
        if self.request_temperature is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TEMPERATURE,
                _value(self.request_temperature),
            )
        if self.request_top_k is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TOP_K,
                _value(self.request_top_k),
            )
        if self.request_top_p is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TOP_P,
                _value(self.request_top_p),
            )
        if self.response_finish_reasons is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_FINISH_REASONS,
                _value(self.response_finish_reasons),
            )
        if self.response_id is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_ID,
                _value(self.response_id),
            )
        if self.response_model is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_MODEL,
                _value(self.response_model),
            )
        if self.response_time_to_first_chunk is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK,
                _value(self.response_time_to_first_chunk),
            )
        if self.usage_audio_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_audio_cache_read_input_tokens),
            )
        if self.usage_audio_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
                _value(self.usage_audio_input_tokens),
            )
        if self.usage_audio_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
                _value(self.usage_audio_output_tokens),
            )
        if self.usage_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_cache_read_input_tokens),
            )
        if self.usage_cache_write_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
                _value(self.usage_cache_write_input_tokens),
            )
        if self.usage_image_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_image_cache_read_input_tokens),
            )
        if self.usage_image_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
                _value(self.usage_image_input_tokens),
            )
        if self.usage_image_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
                _value(self.usage_image_output_tokens),
            )
        if self.usage_input_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_INPUT_TOKENS,
                _value(self.usage_input_tokens),
            )
        if self.usage_output_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_OUTPUT_TOKENS,
                _value(self.usage_output_tokens),
            )
        if self.usage_reasoning_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
                _value(self.usage_reasoning_output_tokens),
            )
        if self.usage_text_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_text_cache_read_input_tokens),
            )
        if self.usage_text_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_INPUT_TOKENS,
                _value(self.usage_text_input_tokens),
            )
        if self.usage_text_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
                _value(self.usage_text_output_tokens),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.input_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_INPUT_MESSAGES,
                    self.input_messages,
                )
            if self.output_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_OUTPUT_MESSAGES,
                    self.output_messages,
                )
            if self.prompt_variable is not None:
                for k, v in self.prompt_variable.items():
                    val = v if isinstance(v, str) else gen_ai_json_dumps(v)
                    span.set_attribute(f"gen_ai.prompt.variable.{k}", val)
            if self.system_instructions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_SYSTEM_INSTRUCTIONS,
                    self.system_instructions,
                )
            if self.tool_definitions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_DEFINITIONS,
                    self.tool_definitions,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_client_operation_duration(duration_s, context=ctx)
        log_record: LogRecord | None = None
        should_emit = emit_event if emit_event is not None else self.emit_event
        if should_emit and self._logger is not None:
            event_attrs: dict[str, AnyValue] = {}
            if self.error_type is not None:
                event_attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
            if self.conversation_compacted:
                event_attrs[Attr.GEN_AI_CONVERSATION_COMPACTED] = _value(
                    self.conversation_compacted
                )
            if self.conversation_id is not None:
                event_attrs[Attr.GEN_AI_CONVERSATION_ID] = _value(
                    self.conversation_id
                )
            if self._operation_name is not None:
                event_attrs[Attr.GEN_AI_OPERATION_NAME] = _value(
                    self._operation_name
                )
            if self.output_type is not None:
                event_attrs[Attr.GEN_AI_OUTPUT_TYPE] = _value(self.output_type)
            if self.prompt_name is not None:
                event_attrs[Attr.GEN_AI_PROMPT_NAME] = _value(self.prompt_name)
            if self.prompt_version is not None:
                event_attrs[Attr.GEN_AI_PROMPT_VERSION] = _value(
                    self.prompt_version
                )
            if self._provider_name is not None:
                event_attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(
                    self._provider_name
                )
            if self.request_choice_count is not None:
                event_attrs[Attr.GEN_AI_REQUEST_CHOICE_COUNT] = _value(
                    self.request_choice_count
                )
            if self.request_frequency_penalty is not None:
                event_attrs[Attr.GEN_AI_REQUEST_FREQUENCY_PENALTY] = _value(
                    self.request_frequency_penalty
                )
            if self.request_max_tokens is not None:
                event_attrs[Attr.GEN_AI_REQUEST_MAX_TOKENS] = _value(
                    self.request_max_tokens
                )
            if self._request_model is not None:
                event_attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(
                    self._request_model
                )
            if self.request_presence_penalty is not None:
                event_attrs[Attr.GEN_AI_REQUEST_PRESENCE_PENALTY] = _value(
                    self.request_presence_penalty
                )
            if self.request_previous_response_id is not None:
                event_attrs[Attr.GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID] = _value(
                    self.request_previous_response_id
                )
            if self.request_reasoning_level is not None:
                event_attrs[Attr.GEN_AI_REQUEST_REASONING_LEVEL] = _value(
                    self.request_reasoning_level
                )
            if self.request_seed is not None:
                event_attrs[Attr.GEN_AI_REQUEST_SEED] = _value(
                    self.request_seed
                )
            if self.request_stop_sequences is not None:
                event_attrs[Attr.GEN_AI_REQUEST_STOP_SEQUENCES] = _value(
                    self.request_stop_sequences
                )
            if self.request_stream is not None:
                event_attrs[Attr.GEN_AI_REQUEST_STREAM] = _value(
                    self.request_stream
                )
            if self.request_temperature is not None:
                event_attrs[Attr.GEN_AI_REQUEST_TEMPERATURE] = _value(
                    self.request_temperature
                )
            if self.request_top_k is not None:
                event_attrs[Attr.GEN_AI_REQUEST_TOP_K] = _value(
                    self.request_top_k
                )
            if self.request_top_p is not None:
                event_attrs[Attr.GEN_AI_REQUEST_TOP_P] = _value(
                    self.request_top_p
                )
            if self.response_finish_reasons is not None:
                event_attrs[Attr.GEN_AI_RESPONSE_FINISH_REASONS] = _value(
                    self.response_finish_reasons
                )
            if self.response_id is not None:
                event_attrs[Attr.GEN_AI_RESPONSE_ID] = _value(self.response_id)
            if self.response_model is not None:
                event_attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(
                    self.response_model
                )
            if self.response_time_to_first_chunk is not None:
                event_attrs[Attr.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK] = _value(
                    self.response_time_to_first_chunk
                )
            if self.usage_audio_cache_read_input_tokens:
                event_attrs[
                    Attr.GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS
                ] = _value(self.usage_audio_cache_read_input_tokens)
            if self.usage_audio_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_AUDIO_INPUT_TOKENS] = _value(
                    self.usage_audio_input_tokens
                )
            if self.usage_audio_output_tokens:
                event_attrs[Attr.GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS] = _value(
                    self.usage_audio_output_tokens
                )
            if self.usage_cache_read_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] = (
                    _value(self.usage_cache_read_input_tokens)
                )
            if self.usage_cache_write_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS] = (
                    _value(self.usage_cache_write_input_tokens)
                )
            if self.usage_image_cache_read_input_tokens:
                event_attrs[
                    Attr.GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS
                ] = _value(self.usage_image_cache_read_input_tokens)
            if self.usage_image_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_IMAGE_INPUT_TOKENS] = _value(
                    self.usage_image_input_tokens
                )
            if self.usage_image_output_tokens:
                event_attrs[Attr.GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS] = _value(
                    self.usage_image_output_tokens
                )
            if self.usage_input_tokens is not None:
                event_attrs[Attr.GEN_AI_USAGE_INPUT_TOKENS] = _value(
                    self.usage_input_tokens
                )
            if self.usage_output_tokens is not None:
                event_attrs[Attr.GEN_AI_USAGE_OUTPUT_TOKENS] = _value(
                    self.usage_output_tokens
                )
            if self.usage_reasoning_output_tokens:
                event_attrs[Attr.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS] = (
                    _value(self.usage_reasoning_output_tokens)
                )
            if self.usage_text_cache_read_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS] = (
                    _value(self.usage_text_cache_read_input_tokens)
                )
            if self.usage_text_input_tokens:
                event_attrs[Attr.GEN_AI_USAGE_TEXT_INPUT_TOKENS] = _value(
                    self.usage_text_input_tokens
                )
            if self.usage_text_output_tokens:
                event_attrs[Attr.GEN_AI_USAGE_TEXT_OUTPUT_TOKENS] = _value(
                    self.usage_text_output_tokens
                )
            if self._server_address is not None:
                event_attrs[ServerAttr.SERVER_ADDRESS] = _value(
                    self._server_address
                )
            if self._server_port is not None:
                event_attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
            if mode in (
                ContentCapturingMode.EVENT_ONLY,
                ContentCapturingMode.SPAN_AND_EVENT,
            ):
                if self.input_messages:
                    event_attrs[Attr.GEN_AI_INPUT_MESSAGES] = (
                        _structured_value(self.input_messages)
                    )
                if self.output_messages:
                    event_attrs[Attr.GEN_AI_OUTPUT_MESSAGES] = (
                        _structured_value(self.output_messages)
                    )
                if self.prompt_variable is not None:
                    for k, v in self.prompt_variable.items():
                        event_attrs[f"gen_ai.prompt.variable.{k}"] = (
                            v if isinstance(v, str) else gen_ai_json_dumps(v)
                        )
                if self.system_instructions:
                    event_attrs[Attr.GEN_AI_SYSTEM_INSTRUCTIONS] = (
                        _structured_value(self.system_instructions)
                    )
                if self.tool_definitions:
                    event_attrs[Attr.GEN_AI_TOOL_DEFINITIONS] = (
                        _structured_value(self.tool_definitions)
                    )
            if self.attributes:
                event_attrs.update(self.attributes)
            log_record = LogRecord(
                event_name="gen_ai.client.inference.operation.details",
                context=ctx,
                attributes=event_attrs,
            )
            self._logger.emit(log_record)
        if self.completion_hook is not None:
            self.completion_hook.on_completion(
                inputs=(
                    list(self.input_messages) if self.input_messages else []
                ),
                outputs=(
                    list(self.output_messages) if self.output_messages else []
                ),
                system_instruction=cast(
                    "list[MessagePart]",
                    list(self.system_instructions)
                    if self.system_instructions
                    else [],
                ),
                tool_definitions=(
                    list(self.tool_definitions)
                    if self.tool_definitions
                    else None
                ),
                span=span,
                log_record=log_record,
            )
        span.end(end_time=end_time)

    def _record_client_operation_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_time_per_output_chunk(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_time_per_output_chunk_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_time_to_first_chunk(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_time_to_first_chunk_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_token_usage(
        self,
        value: int,
        token_type: Attr.GenAITokenType | str,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {
            Attr.GEN_AI_TOKEN_TYPE: _value(token_type),
        }
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self.response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(self.response_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_token_usage_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _client_operation_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_operation_time_per_output_chunk_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.time_per_output_chunk",
            unit="s",
            description="Time per output chunk, recorded for each chunk received after the first one, measured as the time elapsed from the end of the previous chunk to the end of the current chunk.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_operation_time_to_first_chunk_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            description="Time to receive the first chunk, measured from when the client issues the generation request to when the first chunk is received in the response stream.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_token_usage_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used.",
            explicit_bucket_boundaries_advisory=[
                1,
                4,
                16,
                64,
                256,
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                67108864,
            ],
        )


class InvokeAgentClientOperation:
    """`gen_ai.invoke_agent.client` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        completion_hook: CompletionHook | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        agent_description: str | None = None,
        agent_id: str | None = None,
        agent_name: str | None = None,
        agent_version: str | None = None,
        conversation_id: str | None = None,
        data_source_id: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        input_messages: Sequence[InputMessage] | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        output_messages: Sequence[OutputMessage] | None = None,
        output_type: Attr.GenAIOutputType | str | None = None,
        provider_name: Attr.GenAIProviderName | str | None = None,
        request_choice_count: int | None = None,
        request_frequency_penalty: float | None = None,
        request_max_tokens: int | None = None,
        request_model: str | None = None,
        request_presence_penalty: float | None = None,
        request_previous_response_id: str | None = None,
        request_seed: int | None = None,
        request_stop_sequences: Sequence[str] | None = None,
        request_temperature: float | None = None,
        request_top_p: float | None = None,
        response_finish_reasons: Sequence[str] | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        system_instructions: Sequence[SystemInstructionPart] | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        usage_audio_cache_read_input_tokens: int | None = None,
        usage_audio_input_tokens: int | None = None,
        usage_audio_output_tokens: int | None = None,
        usage_cache_read_input_tokens: int | None = None,
        usage_cache_write_input_tokens: int | None = None,
        usage_image_cache_read_input_tokens: int | None = None,
        usage_image_input_tokens: int | None = None,
        usage_image_output_tokens: int | None = None,
        usage_input_tokens: int | None = None,
        usage_output_tokens: int | None = None,
        usage_text_cache_read_input_tokens: int | None = None,
        usage_text_input_tokens: int | None = None,
        usage_text_output_tokens: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self.completion_hook: CompletionHook | None = completion_hook
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.agent_description: str | None = agent_description
        self.agent_id: str | None = agent_id
        self._agent_name: str | None = agent_name
        self.agent_version: str | None = agent_version
        self.conversation_id: str | None = conversation_id
        self.data_source_id: str | None = data_source_id
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self.input_messages: Sequence[InputMessage] | None = input_messages
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.output_messages: Sequence[OutputMessage] | None = output_messages
        self.output_type: Attr.GenAIOutputType | str | None = output_type
        self._provider_name: Attr.GenAIProviderName | str | None = (
            provider_name
        )
        self.request_choice_count: int | None = request_choice_count
        self.request_frequency_penalty: float | None = (
            request_frequency_penalty
        )
        self.request_max_tokens: int | None = request_max_tokens
        self._request_model: str | None = request_model
        self.request_presence_penalty: float | None = request_presence_penalty
        self.request_previous_response_id: str | None = (
            request_previous_response_id
        )
        self.request_seed: int | None = request_seed
        self.request_stop_sequences: Sequence[str] | None = (
            request_stop_sequences
        )
        self.request_temperature: float | None = request_temperature
        self.request_top_p: float | None = request_top_p
        self.response_finish_reasons: Sequence[str] | None = (
            response_finish_reasons
        )
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.system_instructions: Sequence[SystemInstructionPart] | None = (
            system_instructions
        )
        self.tool_definitions: Sequence[ToolDefinition] | None = (
            tool_definitions
        )
        self.usage_audio_cache_read_input_tokens: int | None = (
            usage_audio_cache_read_input_tokens
        )
        self.usage_audio_input_tokens: int | None = usage_audio_input_tokens
        self.usage_audio_output_tokens: int | None = usage_audio_output_tokens
        self.usage_cache_read_input_tokens: int | None = (
            usage_cache_read_input_tokens
        )
        self.usage_cache_write_input_tokens: int | None = (
            usage_cache_write_input_tokens
        )
        self.usage_image_cache_read_input_tokens: int | None = (
            usage_image_cache_read_input_tokens
        )
        self.usage_image_input_tokens: int | None = usage_image_input_tokens
        self.usage_image_output_tokens: int | None = usage_image_output_tokens
        self.usage_input_tokens: int | None = usage_input_tokens
        self.usage_output_tokens: int | None = usage_output_tokens
        self.usage_text_cache_read_input_tokens: int | None = (
            usage_text_cache_read_input_tokens
        )
        self.usage_text_input_tokens: int | None = usage_text_input_tokens
        self.usage_text_output_tokens: int | None = usage_text_output_tokens
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or (
            self.completion_hook is not None
            and not isinstance(self.completion_hook, _NoOpCompletionHook)
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        agent_name = _span_name_value(self._agent_name)
        operation_name = _span_name_value(self._operation_name)
        if operation_name and agent_name:
            return f"{operation_name} {agent_name}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.invoke_agent.client"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            sampling_attributes[Attr.GEN_AI_AGENT_NAME] = _value(
                self._agent_name
            )
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._provider_name is not None:
            sampling_attributes[Attr.GEN_AI_PROVIDER_NAME] = _value(
                self._provider_name
            )
        if self._request_model is not None:
            sampling_attributes[Attr.GEN_AI_REQUEST_MODEL] = _value(
                self._request_model
            )
        if self._server_address is not None:
            sampling_attributes[ServerAttr.SERVER_ADDRESS] = _value(
                self._server_address
            )
        if self._server_port is not None:
            sampling_attributes[ServerAttr.SERVER_PORT] = _value(
                self._server_port
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.agent_description is not None:
            span.set_attribute(
                Attr.GEN_AI_AGENT_DESCRIPTION,
                _value(self.agent_description),
            )
        if self.agent_id is not None:
            span.set_attribute(
                Attr.GEN_AI_AGENT_ID,
                _value(self.agent_id),
            )
        if self.agent_version is not None:
            span.set_attribute(
                Attr.GEN_AI_AGENT_VERSION,
                _value(self.agent_version),
            )
        if self.conversation_id is not None:
            span.set_attribute(
                Attr.GEN_AI_CONVERSATION_ID,
                _value(self.conversation_id),
            )
        if self.data_source_id is not None:
            span.set_attribute(
                Attr.GEN_AI_DATA_SOURCE_ID,
                _value(self.data_source_id),
            )
        if self.output_type is not None:
            span.set_attribute(
                Attr.GEN_AI_OUTPUT_TYPE,
                _value(self.output_type),
            )
        if self.request_choice_count is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_CHOICE_COUNT,
                _value(self.request_choice_count),
            )
        if self.request_frequency_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                _value(self.request_frequency_penalty),
            )
        if self.request_max_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_MAX_TOKENS,
                _value(self.request_max_tokens),
            )
        if self.request_presence_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_PRESENCE_PENALTY,
                _value(self.request_presence_penalty),
            )
        if self.request_previous_response_id is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_PREVIOUS_RESPONSE_ID,
                _value(self.request_previous_response_id),
            )
        if self.request_seed is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_SEED,
                _value(self.request_seed),
            )
        if self.request_stop_sequences is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_STOP_SEQUENCES,
                _value(self.request_stop_sequences),
            )
        if self.request_temperature is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TEMPERATURE,
                _value(self.request_temperature),
            )
        if self.request_top_p is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TOP_P,
                _value(self.request_top_p),
            )
        if self.response_finish_reasons is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_FINISH_REASONS,
                _value(self.response_finish_reasons),
            )
        if self.usage_audio_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_audio_cache_read_input_tokens),
            )
        if self.usage_audio_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_INPUT_TOKENS,
                _value(self.usage_audio_input_tokens),
            )
        if self.usage_audio_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_AUDIO_OUTPUT_TOKENS,
                _value(self.usage_audio_output_tokens),
            )
        if self.usage_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_cache_read_input_tokens),
            )
        if self.usage_cache_write_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_CACHE_WRITE_INPUT_TOKENS,
                _value(self.usage_cache_write_input_tokens),
            )
        if self.usage_image_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_image_cache_read_input_tokens),
            )
        if self.usage_image_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_INPUT_TOKENS,
                _value(self.usage_image_input_tokens),
            )
        if self.usage_image_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_IMAGE_OUTPUT_TOKENS,
                _value(self.usage_image_output_tokens),
            )
        if self.usage_input_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_INPUT_TOKENS,
                _value(self.usage_input_tokens),
            )
        if self.usage_output_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_OUTPUT_TOKENS,
                _value(self.usage_output_tokens),
            )
        if self.usage_text_cache_read_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_CACHE_READ_INPUT_TOKENS,
                _value(self.usage_text_cache_read_input_tokens),
            )
        if self.usage_text_input_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_INPUT_TOKENS,
                _value(self.usage_text_input_tokens),
            )
        if self.usage_text_output_tokens:
            span.set_attribute(
                Attr.GEN_AI_USAGE_TEXT_OUTPUT_TOKENS,
                _value(self.usage_text_output_tokens),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.input_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_INPUT_MESSAGES,
                    self.input_messages,
                )
            if self.output_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_OUTPUT_MESSAGES,
                    self.output_messages,
                )
            if self.system_instructions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_SYSTEM_INSTRUCTIONS,
                    self.system_instructions,
                )
            if self.tool_definitions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_DEFINITIONS,
                    self.tool_definitions,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_client_operation_duration(duration_s, context=ctx)
        if self.completion_hook is not None:
            self.completion_hook.on_completion(
                inputs=(
                    list(self.input_messages) if self.input_messages else []
                ),
                outputs=(
                    list(self.output_messages) if self.output_messages else []
                ),
                system_instruction=cast(
                    "list[MessagePart]",
                    list(self.system_instructions)
                    if self.system_instructions
                    else [],
                ),
                tool_definitions=(
                    list(self.tool_definitions)
                    if self.tool_definitions
                    else None
                ),
                span=span,
                log_record=None,
            )
        span.end(end_time=end_time)

    def _record_client_operation_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_operation_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_token_usage(
        self,
        value: int,
        token_type: Attr.GenAITokenType | str,
        *,
        response_model: str | None = None,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {
            Attr.GEN_AI_TOKEN_TYPE: _value(token_type),
        }
        if response_model is not None:
            attrs[Attr.GEN_AI_RESPONSE_MODEL] = _value(response_model)
        if self._operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self._operation_name)
        if self._provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self._provider_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        if self._server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self._server_address)
        if self._server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self._server_port)
        self._client_token_usage_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _client_operation_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )

    @cached_property
    def _client_token_usage_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used.",
            explicit_bucket_boundaries_advisory=[
                1,
                4,
                16,
                64,
                256,
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                67108864,
            ],
        )


class InvokeAgentInternalOperation:
    """`gen_ai.invoke_agent.internal` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        completion_hook: CompletionHook | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        agent_description: str | None = None,
        agent_name: str | None = None,
        conversation_id: str | None = None,
        data_source_id: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        input_messages: Sequence[InputMessage] | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        output_messages: Sequence[OutputMessage] | None = None,
        output_type: Attr.GenAIOutputType | str | None = None,
        request_choice_count: int | None = None,
        request_frequency_penalty: float | None = None,
        request_max_tokens: int | None = None,
        request_model: str | None = None,
        request_presence_penalty: float | None = None,
        request_seed: int | None = None,
        request_stop_sequences: Sequence[str] | None = None,
        request_temperature: float | None = None,
        request_top_p: float | None = None,
        response_finish_reasons: Sequence[str] | None = None,
        system_instructions: Sequence[SystemInstructionPart] | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        usage_input_tokens: int | None = None,
        usage_output_tokens: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self.completion_hook: CompletionHook | None = completion_hook
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.agent_description: str | None = agent_description
        self._agent_name: str | None = agent_name
        self.conversation_id: str | None = conversation_id
        self.data_source_id: str | None = data_source_id
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self.input_messages: Sequence[InputMessage] | None = input_messages
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.output_messages: Sequence[OutputMessage] | None = output_messages
        self.output_type: Attr.GenAIOutputType | str | None = output_type
        self.request_choice_count: int | None = request_choice_count
        self.request_frequency_penalty: float | None = (
            request_frequency_penalty
        )
        self.request_max_tokens: int | None = request_max_tokens
        self._request_model: str | None = request_model
        self.request_presence_penalty: float | None = request_presence_penalty
        self.request_seed: int | None = request_seed
        self.request_stop_sequences: Sequence[str] | None = (
            request_stop_sequences
        )
        self.request_temperature: float | None = request_temperature
        self.request_top_p: float | None = request_top_p
        self.response_finish_reasons: Sequence[str] | None = (
            response_finish_reasons
        )
        self.system_instructions: Sequence[SystemInstructionPart] | None = (
            system_instructions
        )
        self.tool_definitions: Sequence[ToolDefinition] | None = (
            tool_definitions
        )
        self.usage_input_tokens: int | None = usage_input_tokens
        self.usage_output_tokens: int | None = usage_output_tokens
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or (
            self.completion_hook is not None
            and not isinstance(self.completion_hook, _NoOpCompletionHook)
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        agent_name = _span_name_value(self._agent_name)
        operation_name = _span_name_value(self._operation_name)
        if operation_name and agent_name:
            return f"{operation_name} {agent_name}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.invoke_agent.internal"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            sampling_attributes[Attr.GEN_AI_AGENT_NAME] = _value(
                self._agent_name
            )
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        if self._request_model is not None:
            sampling_attributes[Attr.GEN_AI_REQUEST_MODEL] = _value(
                self._request_model
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.INTERNAL,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.agent_description is not None:
            span.set_attribute(
                Attr.GEN_AI_AGENT_DESCRIPTION,
                _value(self.agent_description),
            )
        if self.conversation_id is not None:
            span.set_attribute(
                Attr.GEN_AI_CONVERSATION_ID,
                _value(self.conversation_id),
            )
        if self.data_source_id is not None:
            span.set_attribute(
                Attr.GEN_AI_DATA_SOURCE_ID,
                _value(self.data_source_id),
            )
        if self.output_type is not None:
            span.set_attribute(
                Attr.GEN_AI_OUTPUT_TYPE,
                _value(self.output_type),
            )
        if self.request_choice_count is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_CHOICE_COUNT,
                _value(self.request_choice_count),
            )
        if self.request_frequency_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_FREQUENCY_PENALTY,
                _value(self.request_frequency_penalty),
            )
        if self.request_max_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_MAX_TOKENS,
                _value(self.request_max_tokens),
            )
        if self.request_presence_penalty is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_PRESENCE_PENALTY,
                _value(self.request_presence_penalty),
            )
        if self.request_seed is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_SEED,
                _value(self.request_seed),
            )
        if self.request_stop_sequences is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_STOP_SEQUENCES,
                _value(self.request_stop_sequences),
            )
        if self.request_temperature is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TEMPERATURE,
                _value(self.request_temperature),
            )
        if self.request_top_p is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_TOP_P,
                _value(self.request_top_p),
            )
        if self.response_finish_reasons is not None:
            span.set_attribute(
                Attr.GEN_AI_RESPONSE_FINISH_REASONS,
                _value(self.response_finish_reasons),
            )
        if self.usage_input_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_INPUT_TOKENS,
                _value(self.usage_input_tokens),
            )
        if self.usage_output_tokens is not None:
            span.set_attribute(
                Attr.GEN_AI_USAGE_OUTPUT_TOKENS,
                _value(self.usage_output_tokens),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.input_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_INPUT_MESSAGES,
                    self.input_messages,
                )
            if self.output_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_OUTPUT_MESSAGES,
                    self.output_messages,
                )
            if self.system_instructions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_SYSTEM_INSTRUCTIONS,
                    self.system_instructions,
                )
            if self.tool_definitions:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_TOOL_DEFINITIONS,
                    self.tool_definitions,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_invoke_agent_duration(duration_s, context=ctx)
        if self.completion_hook is not None:
            self.completion_hook.on_completion(
                inputs=(
                    list(self.input_messages) if self.input_messages else []
                ),
                outputs=(
                    list(self.output_messages) if self.output_messages else []
                ),
                system_instruction=cast(
                    "list[MessagePart]",
                    list(self.system_instructions)
                    if self.system_instructions
                    else [],
                ),
                tool_definitions=(
                    list(self.tool_definitions)
                    if self.tool_definitions
                    else None
                ),
                span=span,
                log_record=None,
            )
        span.end(end_time=end_time)

    def _record_invoke_agent_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self._agent_name is not None:
            attrs[Attr.GEN_AI_AGENT_NAME] = _value(self._agent_name)
        if self._request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self._request_model)
        self._invoke_agent_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_inference_calls(
        self,
        value: int,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            attrs[Attr.GEN_AI_AGENT_NAME] = _value(self._agent_name)
        self._invoke_agent_inference_calls_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    def record_tool_calls(
        self,
        value: int,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self._agent_name is not None:
            attrs[Attr.GEN_AI_AGENT_NAME] = _value(self._agent_name)
        self._invoke_agent_tool_calls_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _invoke_agent_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.invoke_agent.duration",
            unit="s",
            description="The end-to-end duration of a single in-process agent invocation, from the moment the invocation starts until the agent emits the last chunk of its final response or terminates with an error.",
            explicit_bucket_boundaries_advisory=[
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
            ],
        )

    @cached_property
    def _invoke_agent_inference_calls_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.invoke_agent.inference_calls",
            unit="{inference_call}",
            description="The number of inference (model) calls a GenAI agent makes during a single invocation.",
        )

    @cached_property
    def _invoke_agent_tool_calls_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.invoke_agent.tool_calls",
            unit="{tool_call}",
            description="The number of tool calls a GenAI agent makes during a single invocation.",
        )


class InvokeWorkflowInternalOperation:
    """`gen_ai.invoke_workflow.internal` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        completion_hook: CompletionHook | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        conversation_id: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        input_messages: Sequence[InputMessage] | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        output_messages: Sequence[OutputMessage] | None = None,
        workflow_name: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self.completion_hook: CompletionHook | None = completion_hook
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.conversation_id: str | None = conversation_id
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self.input_messages: Sequence[InputMessage] | None = input_messages
        self._operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.output_messages: Sequence[OutputMessage] | None = output_messages
        self.workflow_name: str | None = workflow_name
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or (
            self.completion_hook is not None
            and not isinstance(self.completion_hook, _NoOpCompletionHook)
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        operation_name = _span_name_value(self._operation_name)
        workflow_name = _span_name_value(self.workflow_name)
        if operation_name and workflow_name:
            return f"{operation_name} {workflow_name}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.invoke_workflow.internal"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        if self._operation_name is not None:
            sampling_attributes[Attr.GEN_AI_OPERATION_NAME] = _value(
                self._operation_name
            )
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.INTERNAL,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.conversation_id is not None:
            span.set_attribute(
                Attr.GEN_AI_CONVERSATION_ID,
                _value(self.conversation_id),
            )
        if self.workflow_name is not None:
            span.set_attribute(
                Attr.GEN_AI_WORKFLOW_NAME,
                _value(self.workflow_name),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.input_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_INPUT_MESSAGES,
                    self.input_messages,
                )
            if self.output_messages:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_OUTPUT_MESSAGES,
                    self.output_messages,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_invoke_workflow_duration(duration_s, context=ctx)
        if self.completion_hook is not None:
            self.completion_hook.on_completion(
                inputs=(
                    list(self.input_messages) if self.input_messages else []
                ),
                outputs=(
                    list(self.output_messages) if self.output_messages else []
                ),
                system_instruction=[],
                tool_definitions=None,
                span=span,
                log_record=None,
            )
        span.end(end_time=end_time)

    def _record_invoke_workflow_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self.workflow_name is not None:
            attrs[Attr.GEN_AI_WORKFLOW_NAME] = _value(self.workflow_name)
        self._invoke_workflow_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _invoke_workflow_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.invoke_workflow.duration",
            unit="s",
            description="Records duration of GenAI workflow.",
            explicit_bucket_boundaries_advisory=[
                1,
                5,
                10,
                30,
                60,
                120,
                300,
                600,
                1800,
                3600,
                7200,
            ],
        )


class RetrievalClientOperation:
    """`gen_ai.retrieval.client` operation."""

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger | None = None,
        *,
        error_type_resolver: ErrorTypeResolver | None = None,
        data_source_id: str | None = None,
        error_type: ErrorAttr.ErrorType | str | None = None,
        operation_name: Attr.GenAIOperationName | str | None = None,
        provider_name: Attr.GenAIProviderName | str | None = None,
        request_model: str | None = None,
        retrieval_documents: AnyValue | None = None,
        retrieval_query_text: str | None = None,
        retrieval_top_k: int | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
        metric_attributes: Mapping[str, AttributeValue] | None = None,
        content_capturing_mode: ContentCapturingMode = ContentCapturingMode.SPAN_AND_EVENT,
        emit_event: bool = True,
    ) -> None:
        self._tracer = tracer
        self._meter = meter
        self._logger = logger
        self._error_type_resolver: ErrorTypeResolver | None = (
            error_type_resolver
        )
        self.content_capturing_mode: ContentCapturingMode = (
            content_capturing_mode
        )
        self.emit_event: bool = emit_event
        self._span: Span | None = None
        self._context: Context | None = None
        self._monotonic_start_s: float = timeit.default_timer()
        self._ended: bool = False
        self._scope: AbstractContextManager[Span] | None = None
        self.data_source_id: str | None = data_source_id
        self.error_type: ErrorAttr.ErrorType | str | None = error_type
        self.operation_name: Attr.GenAIOperationName | str | None = (
            operation_name
        )
        self.provider_name: Attr.GenAIProviderName | str | None = provider_name
        self.request_model: str | None = request_model
        self.retrieval_documents: AnyValue | None = retrieval_documents
        self.retrieval_query_text: str | None = retrieval_query_text
        self.retrieval_top_k: int | None = retrieval_top_k
        self.server_address: str | None = server_address
        self.server_port: int | None = server_port
        self.attributes: dict[str, AttributeValue] = (
            dict(attributes) if attributes else {}
        )
        self.metric_attributes: dict[str, AttributeValue] = (
            dict(metric_attributes) if metric_attributes else {}
        )

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this operation."""
        return self.content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )

    @property
    def span(self) -> Span:
        if self._span is None:
            return INVALID_SPAN
        return self._span

    @property
    def context(self) -> Context:
        if self._context is None:
            return Context()
        return self._context

    @property
    def span_name(self) -> str:
        """The span name resolved from templates based on attribute availability."""
        data_source_id = _span_name_value(self.data_source_id)
        operation_name = _span_name_value(self.operation_name)
        if operation_name and data_source_id:
            return f"{operation_name} {data_source_id}"
        if operation_name:
            return f"{operation_name}"
        return "gen_ai.retrieval.client"

    def start(
        self,
        name: str | None = None,
        *,
        context: Context | None = None,
        start_time: int | None = None,
    ) -> Span:
        """Start the span for this operation."""
        if name is None:
            name = self.span_name
        sampling_attributes: dict[str, AttributeValue] = {}
        self._monotonic_start_s = timeit.default_timer()
        self._span = self._tracer.start_span(
            name,
            context=context,
            kind=SpanKind.CLIENT,
            attributes=sampling_attributes,
            start_time=start_time,
        )
        self._context = set_span_in_context(self._span, context)
        return self._span

    def __enter__(self) -> Self:
        if self._span is not None:
            self._scope = use_span(
                self._span,
                end_on_exit=False,
                record_exception=False,
                set_status_on_exception=False,
            )
            self._scope.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        try:
            if exc_val is not None:
                self.finish(error=exc_val)
            else:
                self.finish()
        finally:
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self._scope = None

    def finish(
        self,
        *,
        error: Error | BaseException | None = None,
        duration_s: float | None = None,
        end_time: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        emit_event: bool | None = None,
        context: Context | None = None,
    ) -> None:
        """Finish the operation: apply attributes, record metrics, emit event, and end span."""
        if self._ended or self._span is None:
            return
        self._ended = True
        span = self._span
        ctx = context if context is not None else self._context
        mode = (
            content_capturing_mode
            if content_capturing_mode is not None
            else self.content_capturing_mode
        )

        if error is not None:
            if isinstance(error, BaseException):
                error = Error.from_exception(error, self._error_type_resolver)
            self.error_type = error.type
            span.set_status(Status(StatusCode.ERROR, error.message))
            span.set_attribute(ErrorAttr.ERROR_TYPE, error.type)
        elif self.error_type is not None:
            span.set_status(Status(StatusCode.ERROR))
            span.set_attribute(ErrorAttr.ERROR_TYPE, self.error_type)

        if self.data_source_id is not None:
            span.set_attribute(
                Attr.GEN_AI_DATA_SOURCE_ID,
                _value(self.data_source_id),
            )
        if self.operation_name is not None:
            span.set_attribute(
                Attr.GEN_AI_OPERATION_NAME,
                _value(self.operation_name),
            )
        if self.provider_name is not None:
            span.set_attribute(
                Attr.GEN_AI_PROVIDER_NAME,
                _value(self.provider_name),
            )
        if self.request_model is not None:
            span.set_attribute(
                Attr.GEN_AI_REQUEST_MODEL,
                _value(self.request_model),
            )
        if self.retrieval_top_k is not None:
            span.set_attribute(
                Attr.GEN_AI_RETRIEVAL_TOP_K,
                _value(self.retrieval_top_k),
            )
        if self.server_address is not None:
            span.set_attribute(
                ServerAttr.SERVER_ADDRESS,
                _value(self.server_address),
            )
        if self.server_port is not None:
            span.set_attribute(
                ServerAttr.SERVER_PORT,
                _value(self.server_port),
            )
        if mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            if self.retrieval_documents is not None:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_RETRIEVAL_DOCUMENTS,
                    self.retrieval_documents,
                )
            if self.retrieval_query_text is not None:
                _set_span_json_attribute(
                    span,
                    Attr.GEN_AI_RETRIEVAL_QUERY_TEXT,
                    self.retrieval_query_text,
                )
        if self.attributes:
            for k, v in self.attributes.items():
                span.set_attribute(k, v)

        if duration_s is None:
            duration_s = max(
                timeit.default_timer() - self._monotonic_start_s, 0.0
            )

        self._record_client_operation_duration(duration_s, context=ctx)
        span.end(end_time=end_time)

    def _record_client_operation_duration(
        self,
        value: float,
        *,
        context: Context | None = None,
    ) -> None:
        attrs: dict[str, AttributeValue] = {}
        if self.error_type is not None:
            attrs[ErrorAttr.ERROR_TYPE] = _value(self.error_type)
        if self.operation_name is not None:
            attrs[Attr.GEN_AI_OPERATION_NAME] = _value(self.operation_name)
        if self.provider_name is not None:
            attrs[Attr.GEN_AI_PROVIDER_NAME] = _value(self.provider_name)
        if self.request_model is not None:
            attrs[Attr.GEN_AI_REQUEST_MODEL] = _value(self.request_model)
        if self.server_address is not None:
            attrs[ServerAttr.SERVER_ADDRESS] = _value(self.server_address)
        if self.server_port is not None:
            attrs[ServerAttr.SERVER_PORT] = _value(self.server_port)
        self._client_operation_duration_instrument.record(
            value,
            attributes=_combine_attributes(attrs, self.metric_attributes),
            context=context if context is not None else self._context,
        )

    @cached_property
    def _client_operation_duration_instrument(self):
        return self._meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration.",
            explicit_bucket_boundaries_advisory=[
                0.01,
                0.02,
                0.04,
                0.08,
                0.16,
                0.32,
                0.64,
                1.28,
                2.56,
                5.12,
                10.24,
                20.48,
                40.96,
                81.92,
            ],
        )


__all__ = [
    "EmbeddingsClientOperation",
    "ExecuteToolInternalOperation",
    "FetchResponseClientOperation",
    "InferenceClientOperation",
    "InvokeAgentClientOperation",
    "InvokeAgentInternalOperation",
    "InvokeWorkflowInternalOperation",
    "RetrievalClientOperation",
]
