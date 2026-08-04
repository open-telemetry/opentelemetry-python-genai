# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for classified ``RETRIEVER`` / ``RANKER`` components -> ``RetrievalInvocation``.

Both retrievers and rankers use built-in, local, deterministic Haystack
components (BM25 keyword search, metadata-field ranking) — no network calls,
no VCR cassette needed.
"""

import json

from haystack import Document
from haystack.components.rankers.meta_field import MetaFieldRanker
from haystack.components.retrievers.in_memory.bm25_retriever import (
    InMemoryBM25Retriever,
)
from haystack.document_stores.in_memory.document_store import (
    InMemoryDocumentStore,
)

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def test_bm25_retriever(span_exporter, instrument_with_content):
    store = InMemoryDocumentStore()
    store.write_documents(
        [
            Document(content="Use pip to install Haystack's latest release"),
            Document(content="Argentina won the World Cup in 2022"),
        ]
    )
    retriever = InMemoryBM25Retriever(document_store=store)
    result = retriever.run(query="How do I install Haystack?", top_k=1)
    documents = result["documents"]
    assert len(documents) == 1
    assert "install" in documents[0].content.lower()

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "retrieval"
    attributes = span.attributes or {}
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 1.0
    assert (
        attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT]
        == "How do I install Haystack?"
    )

    retrieved_documents = json.loads(
        attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    )
    assert len(retrieved_documents) == 1
    assert retrieved_documents[0]["id"] == documents[0].id
    assert retrieved_documents[0]["content"] == documents[0].content


def test_bm25_retriever_no_content_capture(
    span_exporter, instrument_no_content
):
    store = InMemoryDocumentStore()
    store.write_documents(
        [Document(content="Use pip to install Haystack's latest release")]
    )
    retriever = InMemoryBM25Retriever(document_store=store)
    retriever.run(query="install", top_k=1)

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes or {}
    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in attributes
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in attributes
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"


def test_meta_field_ranker(span_exporter, instrument_with_content):
    ranker = MetaFieldRanker(meta_field="rating")
    documents = [
        Document(content="low rated", meta={"rating": 1}),
        Document(content="high rated", meta={"rating": 5}),
    ]
    result = ranker.run(documents=documents)
    ranked_documents = result["documents"]
    assert ranked_documents[0].content == "high rated"

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "retrieval"
    attributes = span.attributes or {}
    assert attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"
    retrieved_documents = json.loads(
        attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    )
    assert [doc["content"] for doc in retrieved_documents] == [
        "high rated",
        "low rated",
    ]
