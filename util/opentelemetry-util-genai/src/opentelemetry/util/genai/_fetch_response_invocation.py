# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Final

from opentelemetry._logs import Logger
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
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.types import (
    ErrorTypeResolver,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)
from opentelemetry.util.types import AttributeValue

# TODO: Migrate to gen_ai_attributes constants once available in the semconv
# package. Added to the GenAI semantic conventions in
# https://github.com/open-telemetry/semantic-conventions-genai/pull/353.
_FETCH_RESPONSE_OPERATION_NAME: Final = "fetch_response"
_GEN_AI_REQUEST_STREAM_CURSOR: Final = "gen_ai.request.stream_cursor"
_GEN_AI_RESPONSE_STATUS: Final = "gen_ai.response.status"


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
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        provider: str,
        *,
        response_id: str,
        request_stream: bool | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        error_type_resolver: ErrorTypeResolver | None = None,
    ) -> None:
        """Use handler.fetch_response() rather than calling this directly."""
        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=_FETCH_RESPONSE_OPERATION_NAME,
            # The response identifier is high cardinality, so semconv keeps it
            # out of the span name.
            span_name=_FETCH_RESPONSE_OPERATION_NAME,
            span_kind=SpanKind.CLIENT,
            error_type_resolver=error_type_resolver,
        )
        self._provider: str = provider
        self._response_id: str = response_id
        self._request_stream = request_stream
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.response_model_name: str | None = None
        self.response_status: str | None = None
        self.finish_reasons: list[str] | None = None
        self.stream_cursor: str | None = None
        self.output_messages: list[OutputMessage] = []
        self.system_instruction: list[MessagePart] = []
        self.tool_definitions: list[ToolDefinition] | None = None
        self._start(self._get_start_attributes())

    @property
    def response_id(self) -> str:
        """The identifier of the response being fetched."""
        return self._response_id

    def _get_start_attributes(self) -> dict[str, AttributeValue]:
        """Return attributes known at span creation time."""
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (server_attributes.SERVER_ADDRESS, self._server_address),
            (server_attributes.SERVER_PORT, self._server_port),
        )
        return {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            GenAI.GEN_AI_PROVIDER_NAME: self._provider,
            GenAI.GEN_AI_RESPONSE_ID: self._response_id,
            **(
                {GenAI.GEN_AI_REQUEST_STREAM: self._request_stream}
                if self._request_stream is not None
                else {}
            ),
            **{k: v for k, v in optional_attrs if v is not None},
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        # response_id intentionally excluded — high cardinality.
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_RESPONSE_MODEL, self.response_model_name),
            (server_attributes.SERVER_ADDRESS, self._server_address),
            (server_attributes.SERVER_PORT, self._server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            GenAI.GEN_AI_PROVIDER_NAME: self._provider,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _get_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (_GEN_AI_REQUEST_STREAM_CURSOR, self.stream_cursor),
            (
                GenAI.GEN_AI_RESPONSE_FINISH_REASONS,
                self.finish_reasons or None,
            ),
            (GenAI.GEN_AI_RESPONSE_MODEL, self.response_model_name),
            (_GEN_AI_RESPONSE_STATUS, self.response_status),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
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
            )
        )
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._metrics_recorder.record(self)
        self._call_completion_hook(
            outputs=self.output_messages,
            system_instruction=self.system_instruction,
            tool_definitions=self.tool_definitions,
        )
