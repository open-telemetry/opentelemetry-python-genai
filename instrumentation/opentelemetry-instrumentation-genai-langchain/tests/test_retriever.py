# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for retriever instrumentation.

These tests exercise the full on_retriever_start / on_retriever_end /
on_retriever_error path end-to-end by driving LangChain's retriever
interface against an in-memory fake retriever and verifying the emitted
spans, attributes, and metrics against the semconv spec.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes

# ---------------------------------------------------------------------------
# Fake retriever helpers
# ---------------------------------------------------------------------------


class _FakeRetriever(BaseRetriever):
    """In-memory retriever — no network calls, no embeddings."""

    documents: list[Document] = Field(default_factory=list)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents

    def _get_ls_params(self, **kwargs: Any) -> Any:
        params = super()._get_ls_params(**kwargs)
        params["ls_vector_store_provider"] = "FakeVectorStore"
        return params


class _ErrorRetriever(BaseRetriever):
    """Retriever that always raises."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        raise RuntimeError("retrieval failed")

    def _get_ls_params(self, **kwargs: Any) -> Any:
        params = super()._get_ls_params(**kwargs)
        params["ls_vector_store_provider"] = "FakeVectorStore"
        return params


# ---------------------------------------------------------------------------
# Happy-path span attributes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "capture_content",
    ["SPAN_ONLY", "NO_CONTENT", "SPAN_AND_EVENT", "EVENT_ONLY"],
)
def test_retrieval_span_attributes(
    span_exporter,
    metric_reader,
    tracer_provider,
    meter_provider,
    logger_provider,
    monkeypatch,
    capture_content,
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", capture_content
    )

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )

    docs = [
        Document(page_content="Paris is the capital of France.", id="doc-1"),
        Document(
            page_content="Berlin is the capital of Germany.",
            metadata={"source": "wiki"},
        ),
    ]
    retriever = _FakeRetriever(documents=docs)

    result = retriever.invoke("What is the capital of France?")
    assert len(result) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "retrieval"
    attrs = span.attributes
    assert attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "retrieval"
    assert attrs[gen_ai_attributes.GEN_AI_PROVIDER_NAME] == "FakeVectorStore"

    should_capture = capture_content in ("SPAN_ONLY", "SPAN_AND_EVENT")
    if should_capture:
        assert (
            attrs[gen_ai_attributes.GEN_AI_RETRIEVAL_QUERY_TEXT]
            == "What is the capital of France?"
        )
        docs_attr = attrs[gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS]
        assert docs_attr is not None
        assert "Paris is the capital of France." in docs_attr
        assert "doc-1" in docs_attr
        assert "Berlin is the capital of Germany." in docs_attr
    else:
        assert gen_ai_attributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs
        assert gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS not in attrs

    instrumentor.uninstrument()


def test_retrieval_span_name_without_data_source_id(
    span_exporter, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    retriever = _FakeRetriever(documents=[])
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    assert spans[0].name == "retrieval"


def test_retrieval_span_no_model_when_ls_embedding_model_absent(
    span_exporter, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    retriever = _FakeRetriever(documents=[])
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    assert gen_ai_attributes.GEN_AI_REQUEST_MODEL not in spans[0].attributes


def test_retrieval_span_model_set_when_ls_embedding_model_present(
    span_exporter, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )

    class _RetrieverWithModel(_FakeRetriever):
        def _get_ls_params(self, **kwargs: Any) -> Any:
            params = super()._get_ls_params(**kwargs)
            params["ls_embedding_model"] = "text-embedding-3-small"
            return params

    retriever = _RetrieverWithModel(documents=[])
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    assert (
        spans[0].attributes[gen_ai_attributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-3-small"
    )


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


def test_retrieval_error_span(
    span_exporter, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    retriever = _ErrorRetriever()

    with pytest.raises(RuntimeError, match="retrieval failed"):
        retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "retrieval"
    attrs = span.attributes
    assert attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "retrieval"
    assert attrs[gen_ai_attributes.GEN_AI_PROVIDER_NAME] == "FakeVectorStore"
    assert attrs[error_attributes.ERROR_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Duration metric
# ---------------------------------------------------------------------------


def test_retrieval_duration_metric_emitted(
    span_exporter, metric_reader, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    retriever = _FakeRetriever(documents=[Document(page_content="content")])
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    metrics_data = metric_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) == 1

    all_metrics = resource_metrics[0].scope_metrics[0].metrics
    duration_metrics = [
        m
        for m in all_metrics
        if m.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ]
    assert len(duration_metrics) == 1

    dp = duration_metrics[0].data.data_points
    assert len(dp) == 1
    assert dp[0].sum > 0

    metric_attrs = dp[0].attributes
    assert metric_attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "retrieval"
    assert (
        metric_attrs[gen_ai_attributes.GEN_AI_PROVIDER_NAME]
        == "FakeVectorStore"
    )

    # Exemplar links back to the span
    assert len(dp[0].exemplars) == 1
    assert dp[0].exemplars[0].span_id == span.get_span_context().span_id
    assert dp[0].exemplars[0].trace_id == span.get_span_context().trace_id


def test_retrieval_error_duration_metric_emitted(
    span_exporter, metric_reader, start_instrumentation, monkeypatch
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    retriever = _ErrorRetriever()

    with pytest.raises(RuntimeError):
        retriever.invoke("q")

    metrics_data = metric_reader.get_metrics_data()
    resource_metrics = metrics_data.resource_metrics
    assert len(resource_metrics) == 1

    all_metrics = resource_metrics[0].scope_metrics[0].metrics
    duration_metrics = [
        m
        for m in all_metrics
        if m.name == gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ]
    assert len(duration_metrics) == 1

    dp = duration_metrics[0].data.data_points
    assert len(dp) == 1
    assert dp[0].sum > 0

    metric_attrs = dp[0].attributes
    assert metric_attrs[error_attributes.ERROR_TYPE] == "RuntimeError"
    assert metric_attrs[gen_ai_attributes.GEN_AI_OPERATION_NAME] == "retrieval"


# ---------------------------------------------------------------------------
# Document mapping
# ---------------------------------------------------------------------------


def test_document_id_in_span_content(
    span_exporter,
    tracer_provider,
    meter_provider,
    logger_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    docs = [Document(page_content="text", id="abc-123", metadata={})]
    retriever = _FakeRetriever(documents=docs)
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    docs_attr = spans[0].attributes[
        gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS
    ]
    assert "abc-123" in docs_attr
    instrumentor.uninstrument()


def test_document_without_id_in_span_content(
    span_exporter,
    tracer_provider,
    meter_provider,
    logger_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    docs = [Document(page_content="no id here", metadata={})]
    retriever = _FakeRetriever(documents=docs)
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    docs_attr = spans[0].attributes[
        gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS
    ]
    assert "no id here" in docs_attr
    instrumentor.uninstrument()


def test_document_metadata_not_in_span_content(
    span_exporter,
    tracer_provider,
    meter_provider,
    logger_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    docs = [
        Document(
            page_content="text",
            metadata={"source": "wiki", "score": 0.9},
        )
    ]
    retriever = _FakeRetriever(documents=docs)
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    docs_attr = spans[0].attributes[
        gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS
    ]
    assert "text" in docs_attr
    assert "wiki" not in docs_attr
    assert "0.9" not in docs_attr
    instrumentor.uninstrument()


def test_empty_documents_in_span_content(
    span_exporter,
    tracer_provider,
    meter_provider,
    logger_provider,
    monkeypatch,
):
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    retriever = _FakeRetriever(documents=[])
    retriever.invoke("q")

    spans = span_exporter.get_finished_spans()
    # documents attribute is set but represents an empty list
    docs_attr = spans[0].attributes.get(
        gen_ai_attributes.GEN_AI_RETRIEVAL_DOCUMENTS
    )
    assert docs_attr == "[]"
    instrumentor.uninstrument()
