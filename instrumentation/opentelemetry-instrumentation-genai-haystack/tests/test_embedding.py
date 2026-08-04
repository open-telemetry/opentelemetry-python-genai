# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for classified ``EMBEDDER`` components -> ``EmbeddingInvocation``.

``EmbeddingInvocation`` only carries aggregate request/response metadata
(``dimension_count``, ``input_tokens``, ``response_model_name``) — there is
no util-genai field for per-document embedded text or vectors.
"""

import pytest
from haystack import Document
from haystack.components.embedders.openai_document_embedder import (
    OpenAIDocumentEmbedder,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

OPENAI = GenAIAttributes.GenAiProviderNameValues.OPENAI.value


@pytest.mark.vcr
def test_document_embedder(span_exporter, instrument_with_content):
    embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    documents = [
        Document(content="Argentina won the World Cup in 2022."),
        Document(content="France won the World Cup in 2018."),
    ]
    response = embedder.run(documents=documents)
    embedded_documents = response["documents"]
    assert len(embedded_documents) == 2
    assert embedded_documents[0].embedding is not None

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "embeddings text-embedding-3-small"
    attributes = span.attributes or {}
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "embeddings"
    assert (
        attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-3-small"
    )
    assert attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == OPENAI
    assert attributes[
        GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
    ] == len(embedded_documents[0].embedding)
