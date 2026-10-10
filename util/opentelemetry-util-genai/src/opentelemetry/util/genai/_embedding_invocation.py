# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
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
from opentelemetry.util.genai.utils import ContentCapturingMode
from opentelemetry.util.types import AttributeValue

EMBEDDING_CONTEXT_KEY: Final[str] = "opentelemetry.genai.embedding.context"


@dataclass
class EmbeddingData(_ContextData):
    """Typed data passed from inner embedding invocations to the outer invocation."""

    provider_name: str | None = None
    request_model: str | None = None
    response_model: str | None = None
    server_address: str | None = None
    server_port: int | None = None
    embeddings_dimension_count: int | None = None
    request_encoding_formats: list[str] | None = None
    usage_input_tokens: int | None = None
    attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )
    metric_attributes: dict[str, AttributeValue] = field(
        default_factory=dict[str, AttributeValue]
    )


class EmbeddingInvocation(GenAIInvocation):
    """Represents a single embedding model invocation.

    Use handler.embedding(provider) rather than constructing this directly.
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
        content_capturing_mode: ContentCapturingMode | None = None,
        start_span: bool = True,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        """Use handler.embedding(provider) rather than calling this directly."""
        _operation_name = GenAI.GenAiOperationNameValues.EMBEDDINGS.value
        start_attributes: dict[str, AttributeValue] = {
            k: v
            for k, v in (
                (GenAI.GEN_AI_REQUEST_MODEL, request_model),
                (GenAI.GEN_AI_PROVIDER_NAME, provider),
                (server_attributes.SERVER_ADDRESS, server_address),
                (server_attributes.SERVER_PORT, server_port),
            )
            if v is not None
        }
        self.data: EmbeddingData = EmbeddingData(
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
            span_name=f"{_operation_name} {request_model}"
            if request_model
            else _operation_name,
            span_kind=SpanKind.CLIENT,
            start_attributes=start_attributes,
            content_capturing_mode=content_capturing_mode,
            context=context,
            _attach_to_context=_attach_to_context,
            start_span=start_span,
            attributes=self.data.attributes,
            metric_attributes=self.data.metric_attributes,
            context_key=EMBEDDING_CONTEXT_KEY,
            dataclass_class_object=EmbeddingData,
        )
        self._provider: str = provider
        self._request_model: str | None = request_model
        self._server_address: str | None = server_address
        self._server_port: int | None = server_port
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes

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
    def response_model_name(self) -> str | None:
        return self.data.response_model

    @response_model_name.setter
    def response_model_name(self, value: str | None) -> None:
        self.data.response_model = value

    @property
    def dimension_count(self) -> int | None:
        return self.data.embeddings_dimension_count

    @dimension_count.setter
    def dimension_count(self, value: int | None) -> None:
        self.data.embeddings_dimension_count = value

    @property
    def encoding_formats(self) -> list[str] | None:
        return self.data.request_encoding_formats

    @encoding_formats.setter
    def encoding_formats(self, value: Sequence[str] | None) -> None:
        self.data.request_encoding_formats = (
            list(value) if value is not None else None
        )

    @property
    def input_tokens(self) -> int | None:
        return self.data.usage_input_tokens

    @input_tokens.setter
    def input_tokens(self, value: int | None) -> None:
        self.data.usage_input_tokens = value

    def enrich_from_context(self, data: EmbeddingData) -> None:
        """Enrich invocation attributes from context data published by inner invocations.

        Outer (root) attributes take precedence over inner values.
        """
        self.data.merge(data, overwrite=False)

    def _get_metric_attributes(self) -> dict[str, AttributeValue]:
        optional_attrs = (
            (GenAI.GEN_AI_PROVIDER_NAME, self.data.provider_name),
            (GenAI.GEN_AI_REQUEST_MODEL, self.data.request_model),
            (GenAI.GEN_AI_RESPONSE_MODEL, self.data.response_model),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
        )
        attrs: dict[str, AttributeValue] = {
            GenAI.GEN_AI_OPERATION_NAME: self._operation_name,
            **{k: v for k, v in optional_attrs if v is not None},
        }
        attrs.update(self.metric_attributes)
        return attrs

    def _get_metric_token_counts(self) -> dict[str, int]:
        if self.data.usage_input_tokens is not None:
            return {
                GenAI.GenAiTokenTypeValues.INPUT.value: (
                    self.data.usage_input_tokens
                )
            }
        return {}

    def _apply_finish(self, error: Error | None = None) -> None:
        if error is not None:
            self._apply_error_attributes(error)
        ctx_data = get_value(EMBEDDING_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, EmbeddingData):
            self.enrich_from_context(ctx_data)
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        optional_attrs = (
            (
                GenAI.GEN_AI_PROVIDER_NAME,
                self.data.provider_name or self._provider,
            ),
            (
                GenAI.GEN_AI_REQUEST_MODEL,
                self.data.request_model or self._request_model,
            ),
            (server_attributes.SERVER_ADDRESS, self.data.server_address),
            (server_attributes.SERVER_PORT, self.data.server_port),
            (
                GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
                self.data.embeddings_dimension_count,
            ),
            (
                GenAI.GEN_AI_REQUEST_ENCODING_FORMATS,
                self.data.request_encoding_formats,
            ),
            (GenAI.GEN_AI_RESPONSE_MODEL, self.data.response_model),
            (GenAI.GEN_AI_USAGE_INPUT_TOKENS, self.data.usage_input_tokens),
        )
        attributes: dict[str, AttributeValue] = {
            key: value for key, value in optional_attrs if value is not None
        }
        attributes.update(self.attributes)
        self.span.set_attributes(attributes)
        self._record_client_metrics()


class SuppressedEmbeddingInvocation(EmbeddingInvocation):
    """Represents an embedding invocation running inside an active embedding context.

    Suppresses span creation and metrics. On stop or fail, publishes its
    attributes to the active embedding context.
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
        content_capturing_mode: ContentCapturingMode | None = None,
        context: Context | None = None,
        _attach_to_context: bool = True,
    ) -> None:
        super().__init__(
            tracer,
            instruments,
            logger,
            _NoOpCompletionHook(),
            provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
            start_span=False,
            context=context,
            _attach_to_context=_attach_to_context,
        )

    def publish_to_context(self, data: EmbeddingData) -> None:
        """Publish invocation attributes to the active embedding context."""
        self.data.attributes = self.attributes
        self.data.metric_attributes = self.metric_attributes
        data.merge(self.data, overwrite=True)

    def _finish(self, error: Error | None = None) -> None:
        if self._finished:
            return
        self._finished = True
        ctx_data = get_value(EMBEDDING_CONTEXT_KEY, context=self._span_context)
        if isinstance(ctx_data, EmbeddingData):
            self.publish_to_context(ctx_data)

    def _apply_finish(self, error: Error | None = None) -> None:
        pass
