# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry._logs import Logger
from opentelemetry.metrics import Meter
from opentelemetry.trace import Tracer
from opentelemetry.util.genai._attribute import _Attribute
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.semconv.gen_ai import GenAiOperationName
from opentelemetry.util.genai.semconv.gen_ai._generated import (
    RetrievalClientOperation,
)
from opentelemetry.util.genai.utils import (
    ContentCapturingMode,
    get_content_capturing_mode,
)


class RetrievalInvocation(GenAIInvocation, RetrievalClientOperation):
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

    _data_source_id = _Attribute[str | None]("data_source_id")
    _provider = _Attribute[str | None]("provider_name")
    _request_model = _Attribute[str | None]("request_model")
    _server_address = _Attribute[str | None]("server_address")
    _server_port = _Attribute[int | None]("server_port")
    top_k = _Attribute[int | None]("retrieval_top_k")
    query_text = _Attribute[str | None]("retrieval_query_text")
    documents = _Attribute[Sequence[Mapping[str, Any]] | None](
        "retrieval_documents"
    )

    def __init__(
        self,
        tracer: Tracer,
        meter: Meter,
        logger: Logger,
        *,
        data_source_id: str | None = None,
        provider: str | None = None,
        request_model: str | None = None,
        server_address: str | None = None,
        server_port: int | None = None,
        content_capturing_mode: ContentCapturingMode | None = None,
    ) -> None:
        """Use handler.retrieval() instead of calling this directly."""
        _operation_name = GenAiOperationName.RETRIEVAL.value
        mode = (
            get_content_capturing_mode()
            if content_capturing_mode is None
            else content_capturing_mode
        )
        RetrievalClientOperation.__init__(
            self,
            tracer,
            meter,
            logger,
            operation_name=_operation_name,
            data_source_id=data_source_id,
            provider_name=provider,
            request_model=request_model,
            server_address=server_address,
            server_port=server_port,
            content_capturing_mode=mode,
        )
        GenAIInvocation.__init__(self)
        self.start()
