# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Agno Knowledge instrumentation."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from agno.knowledge.knowledge import Document, Knowledge

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.trace import SpanKind

GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"


def test_knowledge_search_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Knowledge.search with content capture enabled."""
    kb = Knowledge(name="test_kb", max_results=5)
    kb.vector_db = MagicMock()
    kb.vector_db.name = "pgvector_instance"
    kb.vector_db.provider = "pgvector"
    kb.vector_db.search.return_value = [
        Document(
            content="OpenTelemetry is an observability framework.",
            id="doc_1",
            meta_data={"source": "docs"},
            reranking_score=0.95,
        ),
        Document(
            content="Agno is an agent framework.",
            id="doc_2",
            reranking_score=0.88,
        ),
    ]

    docs = kb.search(query="what is OpenTelemetry?", max_results=5)
    assert len(docs) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval test_kb"
    assert span.kind == SpanKind.CLIENT
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "retrieval"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID) == "test_kb"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME) == "pgvector"
    )
    assert span.attributes.get(GEN_AI_RETRIEVAL_TOP_K) == 5
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "what is OpenTelemetry?"
    )

    raw_docs = span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS)
    assert isinstance(raw_docs, str)
    parsed_docs = json.loads(raw_docs)
    assert parsed_docs == [
        {"id": "doc_1", "score": 0.95},
        {"id": "doc_2", "score": 0.88},
    ]


def test_knowledge_search_no_content_capture(
    instrument_agno,
    span_exporter,
) -> None:
    """Test Knowledge.search with content capture disabled."""
    kb = Knowledge(name="private_kb", max_results=3)
    kb.vector_db = MagicMock()
    kb.vector_db.search.return_value = [
        Document(content="Secret content", id="secret_1")
    ]

    docs = kb.search(query="secret query")
    assert len(docs) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval private_kb"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "retrieval"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "private_kb"
    )
    assert span.attributes.get(GEN_AI_RETRIEVAL_TOP_K) == 3
    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in span.attributes
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in span.attributes


def test_knowledge_asearch_content_capture(
    instrument_agno_content_capture,
    span_exporter,
) -> None:
    """Test Knowledge.asearch with content capture enabled."""
    if not hasattr(Knowledge, "asearch"):
        pytest.skip(
            "Knowledge.asearch is not supported in this version of agno"
        )

    kb = Knowledge(name="async_kb")
    kb.vector_db = MagicMock()
    kb.vector_db.async_search = AsyncMock(
        return_value=[
            Document(
                content="Async retrieved doc",
                id="adoc_1",
                reranking_score=0.9,
            )
        ]
    )

    search_fn = kb.asearch

    async def _run() -> list[Document]:
        return await search_fn(query="async query", max_results=2)

    docs = asyncio.run(_run())
    assert len(docs) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval async_kb"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "async_kb"
    )
    assert span.attributes.get(GEN_AI_RETRIEVAL_TOP_K) == 2
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "async query"
    )
    raw_docs = span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS)
    assert isinstance(raw_docs, str)
    assert json.loads(raw_docs) == [{"id": "adoc_1", "score": 0.9}]


def test_knowledge_asearch_no_content_capture(
    instrument_agno,
    span_exporter,
) -> None:
    """Test Knowledge.asearch with content capture disabled."""
    if not hasattr(Knowledge, "asearch"):
        pytest.skip(
            "Knowledge.asearch is not supported in this version of agno"
        )

    kb = Knowledge(name="private_async_kb", max_results=3)
    kb.vector_db = MagicMock()
    kb.vector_db.async_search = AsyncMock(
        return_value=[Document(content="Secret content", id="secret_1")]
    )

    async def _run() -> list[Document]:
        return await kb.asearch(query="secret query")

    docs = asyncio.run(_run())
    assert len(docs) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval private_async_kb"
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == "retrieval"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "private_async_kb"
    )
    assert span.attributes.get(GEN_AI_RETRIEVAL_TOP_K) == 3
    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in span.attributes
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in span.attributes


def test_knowledge_retrieve_delegation(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Knowledge.retrieve delegates to search and creates a retrieval span."""
    if not hasattr(Knowledge, "retrieve"):
        pytest.skip(
            "Knowledge.retrieve is not supported in this version of agno"
        )

    kb = Knowledge(name="retrieve_kb")
    kb.vector_db = MagicMock()
    kb.vector_db.search.return_value = [Document(content="retrieved doc")]

    docs = kb.retrieve("test query")
    assert len(docs) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "retrieval retrieve_kb"


def test_knowledge_aretrieve_delegation(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that Knowledge.aretrieve delegates to asearch and creates a retrieval span."""
    if not hasattr(Knowledge, "aretrieve"):
        pytest.skip(
            "Knowledge.aretrieve is not supported in this version of agno"
        )

    kb = Knowledge(name="aretrieve_kb")
    kb.vector_db = MagicMock()
    kb.vector_db.async_search = AsyncMock(
        return_value=[Document(content="async retrieved doc")]
    )

    async def _run() -> list[Document]:
        return await kb.aretrieve("async test query")

    docs = asyncio.run(_run())
    assert len(docs) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "retrieval aretrieve_kb"


def test_knowledge_search_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that errors in search record error.type on the span."""
    kb = Knowledge(name="error_kb")

    with pytest.raises(TypeError):
        getattr(kb, "search")()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get("error.type") == "TypeError"


def test_knowledge_asearch_error(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that errors in asearch record error.type on the span."""
    if not hasattr(Knowledge, "asearch"):
        pytest.skip(
            "Knowledge.asearch is not supported in this version of agno"
        )
    kb = Knowledge(name="error_async_kb")

    async def _run() -> None:
        with pytest.raises(TypeError):
            await getattr(kb, "asearch")()

    asyncio.run(_run())

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes.get("error.type") == "TypeError"


def test_cancelled_knowledge_asearch_finishes(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that cancellation in asearch records error.type on the span."""
    if not hasattr(Knowledge, "asearch"):
        pytest.skip(
            "Knowledge.asearch is not supported in this version of agno"
        )
    kb = Knowledge(name="cancelled_kb")
    error = asyncio.CancelledError("cancelled")
    kb.vector_db = MagicMock()
    kb.vector_db.async_search = AsyncMock(side_effect=error)

    async def run() -> None:
        with pytest.raises(asyncio.CancelledError) as caught:
            await kb.asearch("query")
        assert caught.value is error
        spans = span_exporter.get_finished_spans()
        assert len(spans) == 1
        assert (
            spans[0].attributes["error.type"]
            == "asyncio.exceptions.CancelledError"
        )

    asyncio.run(run())


def test_base_exception_knowledge_search_finishes(
    instrument_agno,
    span_exporter,
) -> None:
    """Test that BaseException in search records error.type on the span."""
    kb = Knowledge(name="cancelled_kb")
    error = KeyboardInterrupt("interrupted")
    kb.vector_db = MagicMock()
    kb.vector_db.search.side_effect = error

    with pytest.raises(KeyboardInterrupt) as caught:
        kb.search("query")
    assert caught.value is error
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["error.type"] == "KeyboardInterrupt"
