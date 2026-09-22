# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from opentelemetry._logs import Logger
from opentelemetry.context import Context
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.semconv.gen_ai import (
    GenAiOperationName,
    GenAiTokenType,
)
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    EmbeddingsClientOperation,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)


class EmbeddingInvocation(GenAIInvocation, EmbeddingsClientOperation):
    """Represents a single embedding model invocation.

    Use handler.embedding(provider) rather than constructing this directly.
    """

    _provider = _Attribute[str]("provider_name")
    encoding_formats = _Attribute[list[str] | None]("request_encoding_formats")
    input_tokens = _Attribute[int | None]("usage_input_tokens")
    dimension_count = _Attribute[int | None]("embeddings_dimension_count")
    response_model_name = _Attribute[str | None]("response_model")

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        provider: str,
        *,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Use handler.embedding(provider) rather than calling this directly."""
        _operation_name = GenAiOperationName.EMBEDDINGS.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        EmbeddingsClientOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            operation_name=_operation_name,
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        self.start()

    def _on_finish(self, context: Context | None = None) -> None:
        if self.usage_input_tokens is not None:
            self.record_token_usage(
                self.usage_input_tokens,
                token_type=GenAiTokenType.INPUT,
                context=context,
            )
