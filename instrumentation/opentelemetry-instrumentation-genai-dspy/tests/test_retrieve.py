# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy Retrieve instrumentation."""

from __future__ import annotations

import copy
import json
from typing import Any
from unittest.mock import Mock

import dspy
import pytest

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.instrumentation.genai.dspy.patch import (
    _extract_doc_score,
    _set_retrieval_invocation_documents,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import RetrievalInvocation
from opentelemetry.util.genai.types import RetrievalDocument

_GEN_AI_RETRIEVAL_TOP_K = "gen_ai.retrieval.top_k"


class _DummyPassage:
    def __init__(self, text: str) -> None:
        self.long_text = text


class _DummyRM:
    def __init__(
        self,
        data_source_id: str | None = None,
        should_fail: bool = False,
    ) -> None:
        self.data_source_id = data_source_id
        self.should_fail = should_fail

    def __call__(self, query: str, k: int = 3, **kwargs: Any) -> list[Any]:
        if self.should_fail:
            raise RuntimeError("RM retrieval failure")
        return [_DummyPassage(f"Passage {i} for {query}") for i in range(k)]


@pytest.mark.parametrize("k", [0, 1, 3])
def test_sync_retrieve_execution(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    k: int,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=k)
        res = retrieve("What is OpenTelemetry?")
        assert len(res.passages) == k

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "retrieval"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == k
    assert isinstance(attrs.get(_GEN_AI_RETRIEVAL_TOP_K), int)
    assert (
        attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "What is OpenTelemetry?"
    )

    docs_attr = attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)
    assert isinstance(docs_attr, str)
    docs = json.loads(docs_attr)
    assert docs == [{"id": None, "score": None}] * k
    assert res.passages == [
        f"Passage {i} for What is OpenTelemetry?" for i in range(k)
    ]


def test_retrieval_documents_do_not_stringify_passages() -> None:
    class Passage:
        def __str__(self) -> str:
            raise AssertionError("passage text must not be read")

    handler = Mock(spec=TelemetryHandler)
    handler.should_capture_content.return_value = True
    invocation = Mock(spec=RetrievalInvocation)
    _set_retrieval_invocation_documents(handler, invocation, [Passage()])
    assert invocation.documents == [RetrievalDocument()]


def test_extract_doc_score_excludes_booleans() -> None:
    assert _extract_doc_score(True) is None
    assert _extract_doc_score(False) is None
    assert _extract_doc_score(1) == 1.0
    assert _extract_doc_score(0.85) == 0.85
    assert _extract_doc_score("0.95") == 0.95
    assert _extract_doc_score("invalid") is None


def test_retrieve_forward_direct_call(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=3)
        res = retrieve.forward(query="Direct call", k=2)
        assert len(res.passages) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    assert isinstance(attrs.get(_GEN_AI_RETRIEVAL_TOP_K), int)
    assert attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT) == "Direct call"


def test_retrieve_with_positional_k(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=3)
        res = retrieve("Positional k query", 4)
        assert len(res.passages) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 4
    assert isinstance(attrs.get(_GEN_AI_RETRIEVAL_TOP_K), int)
    assert attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT) == "Positional k query"


def test_retrieve_without_content_capture(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ):
        retrieve = dspy.Retrieve(k=2)
        retrieve("Secret query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    assert isinstance(attrs.get(_GEN_AI_RETRIEVAL_TOP_K), int)
    assert GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs
    assert GenAI.GEN_AI_RETRIEVAL_DOCUMENTS not in attrs


def test_retrieve_error_handling(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM(should_fail=True)
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=2)
        with pytest.raises(RuntimeError, match="RM retrieval failure"):
            retrieve("Failing query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"


def test_retrieve_no_rm_loaded(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    dspy.settings.configure(rm=None)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=2)
        with pytest.raises(AssertionError, match="No RM is loaded."):
            retrieve("Query with no RM")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "AssertionError"
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"


def test_retrieve_data_source_id(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM(data_source_id="wiki_collection")
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=2)
        retrieve("Provider query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval wiki_collection"
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_DATA_SOURCE_ID) == "wiki_collection"
    assert GenAI.GEN_AI_PROVIDER_NAME not in attrs


def test_retrieve_copy_and_deepcopy(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retrieve = dspy.Retrieve(k=2)
        copied = copy.copy(retrieve)
        assert copied.k == 2
        copied("Copied query")

        deep_copied = copy.deepcopy(retrieve)
        assert deep_copied.k == 2
        deep_copied("Deep copied query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].name == "retrieval"
    assert spans[1].name == "retrieval"


def test_retrieve_uninstrument(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()
    dspy.settings.configure(rm=rm)

    instrumentor = DSPyInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )

    retrieve = dspy.Retrieve(k=2)
    retrieve("Instrumented query")
    assert len(span_exporter.get_finished_spans()) == 1

    instrumentor.uninstrument()
    span_exporter.clear()

    retrieve("Uninstrumented query")
    assert len(span_exporter.get_finished_spans()) == 0


def test_colbertv2_direct_and_via_retrieve(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy.dsp.colbertv2 as colbert_mod

    from opentelemetry.semconv.attributes import server_attributes

    monkeypatch.setattr(
        colbert_mod,
        "colbertv2_get_request",
        lambda url, query, k: [
            {"long_text": f"Doc {i}", "pid": 100 + i, "score": 0.9 - 0.1 * i}
            for i in range(k)
        ],
    )

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        colbert = dspy.ColBERTv2(url="http://colbert.local", port=8893)
        res = colbert("ColBERT direct query", k=2)
        assert len(res) == 2

        # When ColBERTv2 is configured as dspy.settings.rm and invoked via dspy.Retrieve,
        # only a single retrieval span should be emitted.
        dspy.settings.configure(rm=colbert)
        retrieve = dspy.Retrieve(k=2)
        ret_res = retrieve("ColBERT via Retrieve")
        assert len(ret_res.passages) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2

    direct_span = spans[0]
    direct_attrs = direct_span.attributes or {}
    assert direct_attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert (
        direct_attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "ColBERT direct query"
    )
    assert direct_attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    assert (
        direct_attrs.get(server_attributes.SERVER_ADDRESS) == "colbert.local"
    )
    assert direct_attrs.get(server_attributes.SERVER_PORT) == 8893
    docs = json.loads(str(direct_attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)))
    assert docs == [
        {"id": "100", "score": 0.9},
        {"id": "101", "score": 0.8},
    ]

    retrieve_span = spans[1]
    retrieve_attrs = retrieve_span.attributes or {}
    assert (
        retrieve_attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "ColBERT via Retrieve"
    )
    assert (
        retrieve_attrs.get(server_attributes.SERVER_ADDRESS) == "colbert.local"
    )
    assert retrieve_attrs.get(server_attributes.SERVER_PORT) == 8893


def test_colbertv2_separate_port_malformed_url_and_error(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy.dsp.colbertv2 as colbert_mod

    from opentelemetry.semconv.attributes import (
        error_attributes,
        server_attributes,
    )

    should_fail = False

    def fake_request(url: str, query: str, k: int) -> list[Any]:
        if should_fail:
            raise ConnectionError("ColBERT server unreachable")
        return [
            {"long_text": f"Doc {i}", "pid": i, "score": 0.5} for i in range(k)
        ]

    monkeypatch.setattr(colbert_mod, "colbertv2_get_request", fake_request)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        # 1. Separate .url (without port) and .port attribute
        colbert = dspy.ColBERTv2(url="http://colbert.local")
        colbert.url = "http://colbert.local"
        colbert.port = 8893
        colbert("Separate port query", k=1)

        # 2. Malformed URL does not raise from telemetry
        colbert_malformed = dspy.ColBERTv2(url="http://[invalid-ipv6")
        colbert_malformed.port = "9000"
        colbert_malformed("Malformed url query", k=1)

        # 3. Error path re-raises original exception and marks span as ERROR
        should_fail = True
        with pytest.raises(
            ConnectionError, match="ColBERT server unreachable"
        ):
            colbert("Error query", k=1)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3

    sep_attrs = spans[0].attributes or {}
    assert sep_attrs.get(server_attributes.SERVER_ADDRESS) == "colbert.local"
    assert sep_attrs.get(server_attributes.SERVER_PORT) == 8893

    mal_attrs = spans[1].attributes or {}
    assert mal_attrs.get(server_attributes.SERVER_ADDRESS) is None
    assert mal_attrs.get(server_attributes.SERVER_PORT) == 9000

    err_span = spans[2]
    assert err_span.status.status_code == StatusCode.ERROR
    err_attrs = err_span.attributes or {}
    assert err_attrs.get(error_attributes.ERROR_TYPE) == "ConnectionError"


def test_embeddings_and_embeddings_with_scores_retrieval(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    import numpy as np
    from dspy.retrievers.embeddings import EmbeddingsWithScores

    from opentelemetry.semconv.attributes import error_attributes

    should_fail = False

    def fake_embedder(texts: list[str]) -> np.ndarray:
        if should_fail:
            raise ValueError("Embedding computation failed")
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    corpus = ["First passage", "Second passage", "Third passage"]

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        emb = dspy.Embeddings(
            corpus=corpus,
            embedder=fake_embedder,
            k=2,
            brute_force_threshold=10,
        )
        res = emb("Find passages")
        assert len(res.passages) == 2

        emb_scores = EmbeddingsWithScores(
            corpus=corpus,
            embedder=fake_embedder,
            k=2,
            brute_force_threshold=10,
        )
        res_scores = emb_scores("Find scored passages")
        assert len(res_scores.passages) == 2

        should_fail = True
        with pytest.raises(ValueError, match="Embedding computation failed"):
            emb("Failing embeddings query")
        with pytest.raises(ValueError, match="Embedding computation failed"):
            emb_scores("Failing scored embeddings query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 4

    emb_attrs = spans[0].attributes or {}
    assert emb_attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert emb_attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT) == "Find passages"
    assert emb_attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    emb_docs = json.loads(str(emb_attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)))
    assert len(emb_docs) == 2
    assert emb_docs[0]["id"] is not None
    assert emb_docs[0]["score"] is None

    scores_attrs = spans[1].attributes or {}
    assert (
        scores_attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "Find scored passages"
    )
    assert scores_attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    scores_docs = json.loads(
        str(scores_attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS))
    )
    assert len(scores_docs) == 2
    assert scores_docs[0]["id"] is not None
    assert isinstance(scores_docs[0]["score"], float)

    for err_span in (spans[2], spans[3]):
        assert err_span.status.status_code == StatusCode.ERROR
        err_attrs = err_span.attributes or {}
        assert err_attrs.get(error_attributes.ERROR_TYPE) == "ValueError"


def test_retrieve_subclass_overriding_forward(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    from opentelemetry.semconv.attributes import error_attributes

    class CustomWeaviateLikeRM(dspy.Retrieve):
        def __init__(self, collection_name: str, k: int = 3) -> None:
            super().__init__(k=k)
            self._weaviate_collection_name = collection_name

        def forward(
            self,
            query_or_queries: str | list[str],
            k: int | None = None,
            **kwargs: Any,
        ) -> dspy.Prediction:
            if query_or_queries == "raise_error":
                raise RuntimeError("Weaviate query error")
            eff_k = k if k is not None else self.k
            return dspy.Prediction(
                passages=[f"Doc {i}" for i in range(eff_k)],
                doc_ids=[f"weav-{i}" for i in range(eff_k)],
            )

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        rm = CustomWeaviateLikeRM("articles_v1", k=2)
        res = rm("What is Weaviate?")
        assert len(res.passages) == 2

        with pytest.raises(RuntimeError, match="Weaviate query error"):
            rm("raise_error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2
    span = spans[0]
    assert span.name == "retrieval articles_v1"
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_DATA_SOURCE_ID) == "articles_v1"
    assert attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT) == "What is Weaviate?"
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2
    docs = json.loads(str(attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)))
    assert docs == [
        {"id": "weav-0", "score": None},
        {"id": "weav-1", "score": None},
    ]

    err_span = spans[1]
    assert err_span.status.status_code == StatusCode.ERROR
    err_attrs = err_span.attributes or {}
    assert err_attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"


def test_retrieve_subclass_fallback_to_settings_rm(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    class MyCustomRetrieve(dspy.Retrieve):
        pass

    configured_rm = _DummyRM()
    dspy.settings.configure(rm=configured_rm)

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        retriever = MyCustomRetrieve(k=2)
        res = retriever("Subclass query")
        assert len(res.passages) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT) == "Subclass query"
    assert attrs.get(_GEN_AI_RETRIEVAL_TOP_K) == 2


def test_retrieval_reentrancy_guard(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dspy.dsp.colbertv2 as colbert_mod

    from opentelemetry.instrumentation.genai.dspy.patch import (
        _in_retrieval_invocation,
    )

    observed_contextvar_states: list[bool] = []
    should_fail = False

    def fake_colbert_request(url: str, query: str, k: int) -> list[Any]:
        observed_contextvar_states.append(_in_retrieval_invocation.get())
        if should_fail:
            raise RuntimeError("ColBERT backend failure")
        return [
            {"long_text": f"Doc {i}", "pid": i, "score": 0.9} for i in range(k)
        ]

    monkeypatch.setattr(
        colbert_mod, "colbertv2_get_request", fake_colbert_request
    )

    assert _in_retrieval_invocation.get() is False

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        colbert = dspy.ColBERTv2(url="http://colbert.local", port=8893)
        dspy.settings.configure(rm=colbert)
        retrieve = dspy.Retrieve(k=2)

        # 1. Nested call chain: Retrieve.__call__ -> Retrieve.forward -> ColBERTv2.__call__
        # must emit only 1 span and keep _in_retrieval_invocation True during execution.
        retrieve("First nested query")
        assert observed_contextvar_states == [True]
        assert _in_retrieval_invocation.get() is False
        assert len(span_exporter.get_finished_spans()) == 1

        # 2. Error inside nested call must still reset _in_retrieval_invocation to False.
        should_fail = True
        with pytest.raises(RuntimeError, match="ColBERT backend failure"):
            retrieve("Failing nested query")
        assert observed_contextvar_states == [True, True]
        assert _in_retrieval_invocation.get() is False
        assert len(span_exporter.get_finished_spans()) == 2

        # 3. Subsequent retrieval after error still emits a span normally.
        should_fail = False
        colbert("Direct recovery query", k=1)
        assert observed_contextvar_states == [True, True, True]
        assert _in_retrieval_invocation.get() is False
        assert len(span_exporter.get_finished_spans()) == 3
