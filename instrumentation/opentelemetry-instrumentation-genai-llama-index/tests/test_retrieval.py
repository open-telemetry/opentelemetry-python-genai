# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import cast

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from opentelemetry.instrumentation.genai.llama_index._handler import (
    _retrieval_documents,
    _retrieval_top_k,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode

_GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"


def test_unconvertible_retrieval_results_are_omitted() -> None:
    assert _retrieval_documents([]) == []
    assert _retrieval_documents([object()]) is None


def test_similarity_top_k_getter_failure_is_ignored() -> None:
    class _FailingRetriever:
        @property
        def similarity_top_k(self) -> int:
            raise KeyboardInterrupt

    assert _retrieval_top_k(cast(BaseRetriever, _FailingRetriever())) is None


class _Retriever(BaseRetriever):
    def __init__(self, error: BaseException | None = None) -> None:
        super().__init__()
        self.similarity_top_k = 2
        self._error = error

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._error:
            raise self._error
        return [
            NodeWithScore(
                node=TextNode(id_="doc-1", text="Paris is in France."),
                score=0.9,
            )
        ]


def _span(exporter):
    spans = [s for s in exporter.get_finished_spans() if s.name == "retrieval"]
    assert len(spans) == 1
    return spans[0]


def test_retrieval_captures_documents_and_query(
    span_exporter, instrument_llama_index_with_content
) -> None:
    _Retriever().retrieve("Where is Paris?")
    span = _span(span_exporter)
    assert span.kind == SpanKind.CLIENT
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"
    )
    top_k = span.attributes[_GEN_AI_RETRIEVAL_TOP_K]
    query_text = span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT]
    documents = span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    assert type(top_k) is int
    assert type(query_text) is str
    assert type(documents) is str
    assert top_k == 2
    assert query_text == "Where is Paris?"
    assert json.loads(documents) == [
        {"id": "doc-1", "content": "Paris is in France.", "score": 0.9}
    ]


def test_retrieval_query_bundle_captures_query(
    span_exporter, instrument_llama_index_with_content
) -> None:
    _Retriever().retrieve(QueryBundle(query_str="Where is Paris?"))
    span = _span(span_exporter)
    assert span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT] == (
        "Where is Paris?"
    )


def test_retrieval_omits_content_without_capture(
    span_exporter, instrument_llama_index
) -> None:
    _Retriever().retrieve("Where is Paris?")
    attrs = _span(span_exporter).attributes
    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in attrs


def test_retrieval_uses_snapshotted_capture_setting(
    span_exporter, instrument_llama_index_with_content, monkeypatch
) -> None:
    # Changing the environment after instrumentation must not alter the
    # invocation's content-capture decision.
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    _Retriever().retrieve("Where is Paris?")
    attrs = _span(span_exporter).attributes
    assert attrs[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT] == (
        "Where is Paris?"
    )
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS in attrs


@pytest.mark.asyncio
async def test_async_retrieval_captures_query_and_documents(
    span_exporter, instrument_llama_index_with_content
) -> None:
    result = await _Retriever().aretrieve("Where is Paris?")
    assert result[0].node_id == "doc-1"
    span = _span(span_exporter)
    query_text = span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT]
    documents = span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    assert type(query_text) is str
    assert type(documents) is str
    assert query_text == "Where is Paris?"
    assert json.loads(documents) == [
        {"id": "doc-1", "content": "Paris is in France.", "score": 0.9}
    ]


def test_sync_retrieval_error_is_unchanged(
    span_exporter, instrument_llama_index
) -> None:
    error = ValueError("retrieval failed")
    with pytest.raises(ValueError) as caught:
        _Retriever(error).retrieve("query")
    assert caught.value is error
    span = _span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_async_retrieval_error_is_unchanged(
    span_exporter, instrument_llama_index
) -> None:
    error = ValueError("retrieval failed")
    with pytest.raises(ValueError) as caught:
        await _Retriever(error).aretrieve("query")
    assert caught.value is error
    span = _span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"
