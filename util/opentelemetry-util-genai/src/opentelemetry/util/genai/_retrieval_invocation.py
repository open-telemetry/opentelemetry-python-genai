# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
)
from opentelemetry.util.genai.completion_hook import (
    CompletionHook,
    _NoOpCompletionHook,
)
from opentelemetry.util.genai.types import RetrievalDocument
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    gen_ai_json_dumps,
)
from opentelemetry.util.types import AttributeValue

RETRIEVAL_CONTEXT_KEY: Final[str] = "opentelemetry.genai.retrieval.context"
_GEN_AI_RETRIEVAL_TOP_K: Final = "gen_ai.retrieval.top_k"


@dataclass
class RetrievalData(_ContextData):
    """Typed data passed from inner retrieval invocations to the outer invocation."""

    data_source_id: str | None = None
    provider_name: str | None = None
    request_model: str | None = None
    server_address: str | None = None
    server_port: int | None = None
    retrieval_top_k: int | None = None
    retrieval_query_text: str | None = None
    retrieval_documents: (
        Sequence[RetrievalDocument | Mapping[str, object]] | None
    ) = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


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
    - gen_ai.retrieval.top_k: Maximum number of documents to return (Recommended)
    - server.address: Server address (Recommended)
    - gen_ai.retrieval.documents: Retrieved documents (Opt-In, may contain sensitive data)
    - gen_ai.retrieval.query.text: Query text (Opt-In, may contain sensitive data)
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        data_source_id: str | None = None,
        provider: str | None = None,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        start_span: bool = True,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        """Use handler.retrieval() instead of calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.RETRIEVAL.value
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_DATA_SOURCE_ID, data_source_id),
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
            )
            if v is not None
        }
        self.data: RetrievalData = RetrievalData(
            data_source_id=data_source_id,
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
        )
        super().__init__(
            tracer,
            instruments,
            logger,
            completion_hook,
            operation_name=_operation_name,
            span_name=f"{_operation_name} {data_source_id}"
            if data_source_id
            else _operation_name,
            span_kind=SpanKind.CLIENT,
            start_attributes=start_attributes,
            content_capturing_mode=content_capturing_mode,
            start_span=start_span,
            context=context,
            _attach_to_context=_attach_to_context,
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
            context_key=RETRIEVAL_CONTEXT_KEY,
            dataclass_class_object=RetrievalData,
        )
        self._data_source_id: str | None = data_source_id
        self._provider: str | None = provider
        self._request_model: str | None = request_model
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes

    @property
    def data_source_id(self) -> str | None:
        return self.data.data_source_id

    @property
    def provider(self) -> str | None:
        return self.data.provider_name

    @property
    def request_model(self) -> str | None:
        return self.data.request_model

    @property
    def server_address(self) -> str | None:
        return self.data.server_address

    @property
    def server_port(self) -> int | None:
        return self.data.server_port

    @property
    def top_k(self) -> int | None:
        return self.data.retrieval_top_k

    @top_k.setter
    def top_k(self, value: int | None) -> None:
        self.data.retrieval_top_k = value

    @property
    def query_text(self) -> str | None:
        return self.data.retrieval_query_text

    @query_text.setter
    def query_text(self, value: str | None) -> None:
        self.data.retrieval_query_text = value

    @property
    def documents(
        self,
    ) -> Sequence[RetrievalDocument | Mapping[str, object]] | None:
        """Retrieved document models, captured only in span content modes.

        Passing mappings is deprecated; use ``RetrievalDocument`` instead.
        Legacy mappings are still serialized unchanged.
        """
        return self.data.retrieval_documents

    @documents.setter
    def documents(
        self,
        value: Sequence[RetrievalDocument | Mapping[str, object]] | None,
    ) -> None:
        self.data.retrieval_documents = value

    def enrich_from_context(self, data: RetrievalData) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values. Inner
        invocations never override content capture fields.
        """
        retrieval_query_text = self.data.retrieval_query_text
        retrieval_documents = self.data.retrieval_documents

        self.data.merge(data, overwrite=False)

        self.data.retrieval_query_text = retrieval_query_text
        self.data.retrieval_documents = retrieval_documents

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        # data_source_id intentionally excluded — high cardinality
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_PROVIDER_NAME, self.data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, self.data.request_model),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _get_content_attributes_for_span(self) -> dict[str, AttributeValue]:
        if (
            not self.span.is_recording()
            or not self._should_capture_content_on_span
        ):
            return {}
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (
                GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT,
                self.data.retrieval_query_text,
            ),
            (
                GenAI.GEN_AI_RETRIEVAL_DOCUMENTS,
                gen_ai_json_dumps(self.data.retrieval_documents)
                if self.data.retrieval_documents is not None
                else None,
            ),
        )
        return {k: v for k, v in optional_attrs if v is not None}

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_value(RETRIEVAL_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, RetrievalData):
            self.enrich_from_context(ctx_data)
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        optional_attrs: tuple[tuple[str, AttributeValue | None], ...] = (
            (GenAI.GEN_AI_DATA_SOURCE_ID, self.data.data_source_id),
            (GenAI.GEN_AI_PROVIDER_NAME, self.data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, self.data.request_model),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
            (
                _GEN_AI_RETRIEVAL_TOP_K,
                int(self.data.retrieval_top_k)
                if self.data.retrieval_top_k is not None
                else None,
            ),
        )
        attributes: dict[str, AttributeValue] = {
            k: v for k, v in optional_attrs if v is not None
        }
        attributes.update(self._get_content_attributes_for_span())
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._record_client_metrics()


class SuppressedRetrievalInvocation(RetrievalInvocation):
    """Represents a retrieval invocation running inside an active retrieval context.

    Suppresses span creation and metrics. On stop or fail, publishes its
    attributes to the active retrieval context.
    """

    def __init__(
        self,
        tracer: Tracer,
        instruments: _Instruments,
        logger: Logger,
        completion_hook: CompletionHook,
        *,
        data_source_id: str | None = None,
        provider: str | None = None,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        super().__init__(
            tracer,
            instruments,
            logger,
            _NoOpCompletionHook(),
            data_source_id=data_source_id,
            provider=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
            start_span=False,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def publish_to_context(self, data: RetrievalData) -> None:
        """Publish invocation attributes to the active retrieval context."""
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        data.merge(self.data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        ctx_data = get_value(RETRIEVAL_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, RetrievalData):
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass
