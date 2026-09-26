# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from opentelemetry.instrumentation.genai.llama_index._handler import (
    _retrieval_documents,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode
from opentelemetry.util.genai.types import RetrievalDocument

_GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"


def test_unconvertible_retrieval_results_are_omitted() -> None:
    assert _retrieval_documents([]) == []
    assert _retrieval_documents([object()]) is None


@pytest.mark.parametrize("score", [None, 0.0, 0.9])
def test_retrieval_documents_use_shared_model_without_reading_content(
    score: float | None,
) -> None:
    class LazyTextNode(TextNode):
        def get_content(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("node content must not be read")

    nodes = [NodeWithScore(node=LazyTextNode(id_="doc-1"), score=score)]
    assert _retrieval_documents(nodes) == [
        RetrievalDocument(id="doc-1", score=score)
    ]


class _Retriever(BaseRetriever):
    def __init__(
        self,
        error: BaseException | None = None,
        *,
        nodes: list[NodeWithScore] | None = None,
    ) -> None:
        super().__init__()
        self.similarity_top_k = 2
        self._error = error
        self._nodes = (
            nodes
            if nodes is not None
            else [
                NodeWithScore(
                    node=TextNode(id_="doc-1", text="Paris is in France."),
                    score=0.9,
                )
            ]
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._error:
            raise self._error
        return self._nodes


def _span(exporter):
    spans = [s for s in exporter.get_finished_spans() if s.name == "retrieval"]
    assert len(spans) == 1
    return spans[0]


@pytest.mark.parametrize("attribute", ["node_id", "score"])
@pytest.mark.parametrize("error_type", [RuntimeError, BaseException])
def test_retrieval_documents_skip_broken_accessors(
    attribute: str,
    error_type: type[BaseException],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class BrokenNodeWithScore(NodeWithScore):
        def __getattribute__(self, name: str) -> object:
            if name == attribute:
                raise error_type("broken document accessor")
            return super().__getattribute__(name)

    broken = BrokenNodeWithScore(node=TextNode(id_="broken"), score=0.5)
    valid = NodeWithScore(node=TextNode(id_="valid"), score=0.0)
    with caplog.at_level(logging.WARNING):
        assert _retrieval_documents([valid, broken, valid]) == [
            RetrievalDocument(id="valid", score=0.0),
            RetrievalDocument(id="valid", score=0.0),
        ]
        assert _retrieval_documents([broken]) is None
    assert len(caplog.records) == 2
    assert all(
        record.message == "Failed to extract retrieval document attributes"
        for record in caplog.records
    )


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_broken_document_does_not_change_retrieval_result(
    span_exporter, instrument_llama_index_with_content, is_async: bool
) -> None:
    class BrokenNodeWithScore(NodeWithScore):
        @property
        def node_id(self) -> str:
            raise RuntimeError("broken document ID")

    broken = BrokenNodeWithScore(node=TextNode(id_="broken"), score=0.5)
    valid = NodeWithScore(node=TextNode(id_="valid"), score=0.9)
    retriever = _Retriever(nodes=[broken, valid])
    result = (
        await retriever.aretrieve("query")
        if is_async
        else retriever.retrieve("query")
    )

    assert len(result) == 2
    assert result[0] is broken
    assert result[1] is valid
    span = _span(span_exporter)
    assert span.status.status_code == StatusCode.UNSET
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert json.loads(
        span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    ) == [{"id": "valid", "score": 0.9}]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "score, expected",
    [
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        (None, None),
        (0.0, 0.0),
        (-0.5, -0.5),
        (0.9, 0.9),
    ],
)
@pytest.mark.asyncio
async def test_retrieval_scores_serialize_as_valid_json(
    span_exporter,
    instrument_llama_index_with_content,
    is_async: bool,
    score: float | None,
    expected: float | None,
) -> None:
    node = NodeWithScore(node=TextNode(id_="doc-1"), score=score)
    retriever = _Retriever(nodes=[node])
    result = (
        await retriever.aretrieve("query")
        if is_async
        else retriever.retrieve("query")
    )
    assert result[0] is node
    span = _span(span_exporter)
    raw = span.attributes[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    assert type(raw) is str
    documents = json.loads(
        raw,
        parse_constant=lambda value: pytest.fail(
            f"Non-standard JSON constant: {value}"
        ),
    )
    assert documents == [{"id": "doc-1", "score": expected}]
    if expected is not None:
        assert type(documents[0]["score"]) is float
    assert span.status.status_code == StatusCode.UNSET


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
    assert json.loads(documents) == [{"id": "doc-1", "score": 0.9}]
    assert type(json.loads(documents)[0]["id"]) is str
    assert type(json.loads(documents)[0]["score"]) is float


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
    assert json.loads(documents) == [{"id": "doc-1", "score": 0.9}]


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
