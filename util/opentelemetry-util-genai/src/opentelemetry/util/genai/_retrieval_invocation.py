# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from opentelemetry._logs import Logger
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.trace import SpanKind, Tracer
from opentelemetry.util.genai._invocation import Error, GenAIInvocation
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.metrics import InvocationMetricsRecorder
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
    get_content_capturing_mode,
    is_experimental_mode,
)
from opentelemetry.util.types import AttributeValue


class RetrievalInvocation(GenAIInvocation):
    """Represents a single retrieval invocation (retrieval span).

    Use handler.retrieval() rather than constructing this directly.

    Reference: https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md#retrievals

    Semantic convention attributes for retrieval spans:
    - gen_ai.operation.name: "retrieval" (Required)
    - error.type: Error type if operation failed (Conditionally Required)
    - gen_ai.data_source.id: Data source identifier (Conditionally Required, when applicable)
    - gen_ai.provider.name: Provider name (Conditionally Required, when applicable)
    - gen_ai.request.model: Model name if applicable (Conditionally Required, if available)
    - server.port: Server port (Conditionally Required, if server.address is set)
    - gen_ai.request.top_k: Top-k sampling setting (Recommended)
    - server.address: Server address (Recommended)
    - gen_ai.retrieval.documents: Retrieved documents (Opt-In, may contain sensitive data)
    - gen_ai.retrieval.query.text: Query text (Opt-In, may contain sensitive data)
    """

    def __init__(
        self,
        tracer: Tracer,
        metrics_recorder: InvocationMetricsRecorder,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        data_source_id: str | None = None,
        provider: str | None = None,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
    ) -> None:
        """Use handler.retrieval() instead of calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.RETRIEVAL.value
        super().__init__(
            tracer,
            metrics_recorder,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {data_source_id}"
            if data_source_id
            else _operation_name,
            span_kind=SpanKind.CLIENT,
        )
        self.data_source_id = data_source_id
        self.provider = provider
        self.request_model = request_model
        self.server_address = server_address
        self.server_port = server_port
        self.top_k: float | None = None
        self.query_text: str | None = None
        self.documents: Sequence[Mapping[str, Any]] | None = None
        self._start(self._get_base_attributes())

    def _get_base_attributes(self) -> dict[str, AttributeValue]:
        """Return sampling-relevant attributes available at span creation time."""
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_DATA_SOURCE_ID, self.data_source_id),
            (GenAI.GEN_AI_PROVIDER_NAME, self.provider),
            (GenAI.GEN_AI_REQUEST_MODEL, self.request_model),
            (server_attributes.SERVER_ADDRESS, self.server_address),
            (server_attributes.SERVER_PORT, self.server_port),
        )
        return {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        # data_source_id intentionally excluded — high cardinality
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_PROVIDER_NAME, self.provider),
            (GenAI.GEN_AI_REQUEST_MODEL, self.request_model),
            (server_attributes.SERVER_ADDRESS, self.server_address),
            (server_attributes.SERVER_PORT, self.server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        # TODO: remove cast once base class metric_attributes is typed as dict[str, AttributeValue]
        attrs.update(cast(dict[str, AttributeValue], self.metric_attributes))
        return attrs

    def _get_content_attributes_for_span(self) -> dict[str, AttributeValue]:
        if not self.span.is_recording():
            return {}
        if not is_experimental_mode() or get_content_capturing_mode() not in (
            ContentCapturingMode.SPAN_ONLY,
            ContentCapturingMode.SPAN_AND_EVENT,
        ):
            return {}
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT, self.query_text),
            (
                GenAI.GEN_AI_RETRIEVAL_DOCUMENTS,
                gen_ai_json_dumps(self.documents)
                if self.documents is not None
                else None,
            ),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        attributes: dict[str, AttributeValue] = {}
        if self.top_k is not None:
            attributes[GenAI.GEN_AI_REQUEST_TOP_K] = self.top_k
        attributes.update(self._get_content_attributes_for_span())
        # TODO: remove cast once base class self.attributes is typed as dict[str, AttributeValue]
        attributes.update(cast(dict[str, AttributeValue], self.attributes))
        self.span.set_attributes(attributes)
        self._metrics_recorder.record(self)
