# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import timeit
from abc import abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import Token
from dataclasses import asdict
from types import TracebackType
from typing import Any, TypeAlias, cast

from typing_extensions import Self

from opentelemetry._logs import Logger, LogRecord
from opentelemetry.context import Context, attach, detach
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import Span, SpanKind, Tracer, set_span_in_context
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.genai._conversation_context import (
    get_ambient_conversation_id,
    with_conversation_id,
)
from opentelemetry.util.genai._instruments import _Instruments
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.types import (
    Error,
    ErrorTypeResolver,
    InputMessage,
    MessagePart,
    OutputMessage,
    SystemInstructionPart,
    ToolDefinition,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
)
from opentelemetry.util.types import AttributeValue

_logger = logging.getLogger(__name__)

_GEN_AI_PROMPT_VARIABLE_PREFIX: str = "gen_ai.prompt.variable."


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
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        operation_name: str,
        span_name: str,
        span_kind: SpanKind = SpanKind.CLIENT,
        attributes: dict[str, AttributeValue] | None = None,
        metric_attributes: dict[str, AttributeValue] | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
        *,
        start_attributes: dict[str, AttributeValue] | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
        conversation_id: str | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        self._tracer = tracer
        self._instruments: _Instruments = instruments
        self._logger = logger
        self._completion_hook = completion_hook
        self._error_type_resolver = error_type_resolver
        self._operation_name: str = operation_name
        self._content_capturing_mode: ContentCapturingMode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        self.attributes: dict[str, AttributeValue] = (
            {} if attributes is None else attributes
        )
        """Additional attributes to set on spans and/or events. Not set on metrics."""
        self.metric_attributes: dict[str, AttributeValue] = (
            {} if metric_attributes is None else metric_attributes
        )
        """Additional attributes to set on metrics. Must be low cardinality. Not set on spans or events."""
        self.conversation_id: str | None = (
            conversation_id
            if conversation_id is not None
            else get_ambient_conversation_id(context)
        )
        """Emitted as ``gen_ai.conversation.id`` by the operations semconv
        defines it on: inference, invoke_agent, invoke_workflow."""
        if self.conversation_id:
            context = with_conversation_id(
                self.conversation_id, context=context
            )
        self._start_attributes: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: operation_name,
            **(start_attributes or {}),
        }
        self.span: Span = self._tracer.start_span(
            name=span_name,
            kind=span_kind,
            attributes=self._start_attributes,
            context=context,
        )
        self._span_context: Context = set_span_in_context(self.span, context)
        self._context_token: ContextToken | None = (
            attach(self._span_context) if _attach_to_context else None
        )
        self._monotonic_start_s: float = timeit.default_timer()
        # Streaming state, set when the invocation is handed to a stream
        # wrapper. ``_request_stream`` marks the request as streamed
        # (gen_ai.request.stream); the timing fields are populated by
        # ``_on_stream_chunk`` as each chunk arrives.
        self._request_stream: bool | None = None
        self._ttfc_seconds: float | None = None
        self._stream_last_chunk_at: float | None = None
        self._finished: bool = False

    @property
    def should_capture_content(self) -> bool:
        """Return True when message content should be captured for this invocation."""
        return self._content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.EVENT_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ) or not isinstance(self._completion_hook, _NoOpCompletionHook)

    @property
    def _should_capture_content_on_span(self) -> bool:
        return self._content_capturing_mode in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        )

    @property
    def context(self) -> Context:
        """The OpenTelemetry Context containing this invocation's span."""
        return self._span_context

    def suspend(self) -> None:
        """Restore the context that was current before this invocation started.

        Call this when handing control back to the caller while the invocation
        is still running -- returning a stream the caller has not drained yet,
        for example -- so unrelated caller work is not parented under this
        invocation's span. Idempotent, and pairs with ``activate``.
        """
        token, self._context_token = self._context_token, None
        if token is None:
            return
        # ``attach`` is typed as returning a contextvars token, but a runtime
        # context selected with OTEL_PYTHON_CONTEXT hands out its own token
        # type; only that backend knows how to restore it.
        if not isinstance(token, Token):  # pyright: ignore[reportUnnecessaryIsInstance]
            detach(token)
            return
        # Same reset the contextvars runtime does, without the ERROR log that
        # ``opentelemetry.context.detach`` emits for a token from another context
        # (an invocation finished in a different task). That context is not ours
        # to restore, so a foreign token is a no-op.
        try:
            token.var.reset(token)
        except ValueError:
            _logger.debug(
                "Invocation finished in a different context than it started in;"
                " leaving that context untouched."
            )

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this invocation's span the current span inside the block.

        Restores the previous context on exit. A no-op once the invocation has finished.
        """
        if self._finished:
            yield
            return
        token = attach(self.context)
        try:
            yield
        finally:
            detach(token)

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        """Return low-cardinality attributes for metric recording."""
        return self.metric_attributes

    def _get_metric_token_counts(self) -> dict[str, int]:  # pylint: disable=no-self-use
        """Return {token_type: count} for token histogram recording."""
        return {}

    def record_stream_chunk(self) -> None:
        """Mark the request as streamed and record one output chunk arriving."""
        if self._finished:
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
            self._instruments.time_to_first_chunk.record(
                delta,
                attributes=attributes,
                context=self._span_context,
            )
        else:
            self._instruments.time_per_output_chunk.record(
                delta,
                attributes=attributes,
                context=self._span_context,
            )

    def _record_client_metrics(self) -> None:
        """Record gen_ai.client.operation.duration and gen_ai.client.token.usage."""
        attributes = self._get_metric_attributes()
        duration_seconds = max(
            timeit.default_timer() - self._monotonic_start_s,
            0.0,
        )
        self._instruments.operation_duration.record(
            duration_seconds,
            attributes=attributes,
            context=self._span_context,
        )

        token_counts = self._get_metric_token_counts()
        if token_counts:
            for token_type, token_count in token_counts.items():
                self._instruments.token_usage.record(
                    token_count,
                    attributes=attributes
                    | {GenAI.GEN_AI_TOKEN_TYPE: token_type},
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
        system_instruction: list[SystemInstructionPart]
        | list[MessagePart]
        | None = None,
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
            system_instruction=cast(
                "list[MessagePart]", system_instruction or []
            ),
            tool_definitions=tool_definitions,
            span=self.span,
            log_record=log_record,
        )

    @abstractmethod
    def _apply_finish(self, error: Error | None = None) -> None:
        """Apply finish telemetry (attributes, metrics, events)."""

    def _finish(self, error: Error | None = None) -> None:
        """Apply finish telemetry and end the span. Finishes at most once."""
        if self._finished:
            return
        # Set up front so a nested or repeated finish is a no-op even if
        # _apply_finish raises.
        self._finished = True
        try:
            self._apply_finish(error)
        finally:
            self.suspend()
            self.span.end()

    def stop(self) -> None:
        """Finalize the invocation successfully and end its span."""
        self._finish()

    def fail(self, error: Error | BaseException) -> None:
        """Fail the invocation and end its span with error status."""
        if isinstance(error, BaseException):
            error = Error.from_exception(error, self._error_type_resolver)
        self._finish(error)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_value is not None:
            self.fail(exc_value)
        else:
            self.stop()


def get_content_attributes(
    *,
    input_messages: Sequence[InputMessage],
    output_messages: Sequence[OutputMessage],
    system_instruction: Sequence[SystemInstructionPart | MessagePart],
    tool_definitions: Sequence[ToolDefinition] | None,
    prompt_variables: Mapping[str, object] | None = None,
    for_span: bool,
    content_capturing_mode: ContentCapturingMode | None = None,
) -> dict[str, Any]:
    """Serialize messages, system instructions, and tool definitions into attributes.

    Args:
        input_messages: Input messages to serialize.
        output_messages: Output messages to serialize.
        system_instruction: System instructions to serialize. Passing ``MessagePart``
            is deprecated; use ``SystemInstructionPart``.
        tool_definitions: Tool definitions to serialize (may be None).
        prompt_variables: Prompt template variables to serialize (may be None).
        for_span: If True, serialize for span attributes (JSON string);
                  if False, serialize for event attributes (list of dicts).
        content_capturing_mode: Configured content capturing mode; if None,
                                reads from environment.
    """
    mode = (
        get_content_capturing_mode()
        if content_capturing_mode is None
        else content_capturing_mode
    )
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

    if mode not in allowed_modes:
        return {}

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
    result = {key: value for key, value in optional_attrs if value is not None}
    if prompt_variables:
        for k, v in prompt_variables.items():
            result[f"{_GEN_AI_PROMPT_VARIABLE_PREFIX}{k}"] = (
                v if isinstance(v, str) else gen_ai_json_dumps(v)
            )
    return result
