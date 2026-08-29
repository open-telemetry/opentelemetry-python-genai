# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import timeit
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import AbstractContextManager
from contextvars import Token
from dataclasses import asdict
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeAlias

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import Context, attach, detach
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import INVALID_SPAN as _INVALID_SPAN
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    Error,
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
)
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from opentelemetry.util.genai.metrics import InvocationMetricsRecorder


ContextToken: TypeAlias = Token[Context]


class GenAIInvocation(AbstractContextManager["GenAIInvocation"]):
    """
    Base class for all GenAI invocation types. Manages the lifecycle of a single
    GenAI operation (LLM call, embedding, tool execution, workflow, etc.).

    Use the factory methods on TelemetryHandler (inference, embedding,
    workflow, tool) rather than constructing invocations directly.
    """

    def __init__(
        self,
        # Individual components instead of TelemetryHandler to avoid a circular
        # import between handler.py and the invocation modules.
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        operation_name: str,
        span_name: str,
        span_kind: SpanKind = SpanKind.CLIENT,
        attributes: dict[str, AttributeValue] | None = None,
        metric_attributes: dict[str, AttributeValue] | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
    ) -> None:
        self._tracer = tracer
        self._metrics_recorder = metrics_recorder
        self._logger = logger
        self._completion_hook = completion_hook
        self._error_type_resolver = error_type_resolver
        self._operation_name: str = operation_name
        self.attributes: dict[str, AttributeValue] = (
            {} if attributes is None else attributes
        )
        """Additional attributes to set on spans and/or events. Not set on metrics."""
        self.metric_attributes: dict[str, AttributeValue] = (
            {} if metric_attributes is None else metric_attributes
        )
        """Additional attributes to set on metrics. Must be low cardinality. Not set on spans or events."""
        self.span: Span = _INVALID_SPAN
        self._span_context: Context
        self._span_name: str = span_name
        self._span_kind: SpanKind = span_kind
        self._context_token: ContextToken | None = None
        self._monotonic_start_s: float
        # Streaming state, set when the invocation is handed to a stream
        # wrapper. ``_request_stream`` marks the request as streamed
        # (gen_ai.request.stream); the timing fields are populated by
        # ``_on_stream_chunk`` as each chunk arrives.
        self._request_stream: bool | None = None
        self._ttfc_seconds: float | None = None
        self._stream_last_chunk_at: float | None = None

    def _start(
        self, attributes: dict[str, AttributeValue] | None = None
    ) -> None:
        """Start the invocation span and attach it to the current context.

        Args:
            attributes: Initial span attributes available for sampling decisions.
        """
        self.span = self._tracer.start_span(
            name=self._span_name,
            kind=self._span_kind,
            attributes=attributes,
        )
        self._span_context = set_span_in_context(self.span)
        self._monotonic_start_s = timeit.default_timer()
        self._context_token = attach(self._span_context)

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        """Return low-cardinality attributes for metric recording."""
        return self.metric_attributes

    def _get_metric_token_counts(self) -> dict[str, int]:  # pylint: disable=no-self-use
        """Return {token_type: count} for token histogram recording."""
        return {}

    def record_stream_chunk(self) -> None:
        """Mark the request as streamed and record one output chunk arriving."""
        if self._context_token is None:
            return
        self._request_stream = True
        self._on_stream_chunk(timeit.default_timer())

    def _on_stream_chunk(self, chunk_at: float) -> None:
        """Record streaming timing for one output chunk as it arrives.

        The first chunk's delta from the invocation start is the
        time-to-first-chunk; each later chunk's delta from the previous one is
        the inter-chunk gap. Called by the stream wrapper for any invocation
        type handed to it.
        """
        last_chunk_at = (
            self._stream_last_chunk_at
            if self._stream_last_chunk_at is not None
            else self._monotonic_start_s
        )

        self._stream_last_chunk_at = chunk_at
        delta = max(chunk_at - last_chunk_at, 0.0)
        attributes = self._get_metric_attributes()
        if self._ttfc_seconds is None:
            self._ttfc_seconds = delta
            self._metrics_recorder.record_time_to_first_chunk(
                delta,
                attributes=attributes,
                context=self._span_context,
            )
        else:
            self._metrics_recorder.record_time_per_chunk(
                delta,
                attributes=attributes,
                context=self._span_context,
            )

    def _apply_error_attributes(self, error: Error) -> None:
        """Apply error status and error.type attribute to the span, events, and metrics."""
        self.span.set_status(Status(StatusCode.ERROR, error.message))
        self.attributes[error_attributes.ERROR_TYPE] = error.type
        self.metric_attributes[error_attributes.ERROR_TYPE] = error.type

    def _call_completion_hook(
        self,
        *,
        inputs: list[InputMessage] | None = None,
        outputs: list[OutputMessage] | None = None,
        system_instruction: list[MessagePart] | None = None,
        tool_definitions: list[ToolDefinition] | None = None,
        log_record: LogRecord | None = None,
    ) -> None:
        """Invoke the completion hook with the invocation's content.

        Subclasses pass whichever content fields they carry; the wrapper substitutes []
        for unspecified list fields
        """
        self._completion_hook.on_completion(
            inputs=inputs or [],
            outputs=outputs or [],
            system_instruction=system_instruction or [],
            tool_definitions=tool_definitions,
            span=self.span,
            log_record=log_record,
        )

    @abstractmethod
    def _apply_finish(self, error: Error | None = None) -> None:
        """Apply finish telemetry (attributes, metrics, events)."""

    def _finish(self, error: Error | None = None) -> None:
        """Apply finish telemetry and end the span. Finishes at most once."""
        if self._context_token is None:
            return
        # Clear up front so a nested or repeated finish is a no-op even if
        # _apply_finish raises.
        context_token, self._context_token = self._context_token, None
        try:
            self._apply_finish(error)
        finally:
            try:
                detach(context_token)
            except Exception:  # pylint: disable=broad-except
                pass
            self.span.end()

    def stop(self) -> None:
        """Finalize the invocation successfully and end its span."""
        self._finish()

    def fail(self, error: Error | BaseException) -> None:
        """Fail the invocation and end its span with error status."""
        if isinstance(error, BaseException):
            error = Error.from_exception(error, self._error_type_resolver)
        self._finish(error)

    def __enter__(self) -> GenAIInvocation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None and isinstance(exc_value, Exception):
            self.fail(exc_value)
        else:
            self.stop()


def get_content_attributes(
    *,
    input_messages: Sequence[InputMessage],
    output_messages: Sequence[OutputMessage],
    system_instruction: Sequence[MessagePart],
    tool_definitions: Sequence[ToolDefinition] | None,
    for_span: bool,
) -> dict[str, Any]:
    """Serialize messages, system instructions, and tool definitions into attributes.

    Args:
        input_messages: Input messages to serialize.
        output_messages: Output messages to serialize.
        system_instruction: System instructions to serialize.
        tool_definitions: Tool definitions to serialize (may be None).
        for_span: If True, serialize for span attributes (JSON string);
                  if False, serialize for event attributes (list of dicts).
    """
    mode = get_content_capturing_mode()
    allowed_modes = (
        (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
        if for_span
        else (
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )
    )

    def serialize(items: Sequence[Any]) -> Any:
        dicts = [asdict(item) for item in items]
        return gen_ai_json_dumps(dicts) if for_span else dicts

    # Tool definitions are always captured, the sem conv recommends adding params / description only
    # when the content capture mode is set..
    if mode not in allowed_modes:
        return (
            {GenAI.GEN_AI_TOOL_DEFINITIONS: serialize(tool_definitions)}
            if tool_definitions
            else {}
        )

    optional_attrs = (
        (
            GenAI.GEN_AI_INPUT_MESSAGES,
            serialize(input_messages) if input_messages else None,
        ),
        (
            GenAI.GEN_AI_OUTPUT_MESSAGES,
            serialize(output_messages) if output_messages else None,
        ),
        (
            GenAI.GEN_AI_SYSTEM_INSTRUCTIONS,
            serialize(system_instruction) if system_instruction else None,
        ),
        (
            GenAI.GEN_AI_TOOL_DEFINITIONS,
            serialize(tool_definitions) if tool_definitions else None,
        ),
    )
    return {key: value for key, value in optional_attrs if value is not None}
