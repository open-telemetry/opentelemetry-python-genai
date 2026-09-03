# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy Retrieve instrumentation."""

from __future__ import annotations

import copy
import json
from typing import Any

import dspy
import pytest

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
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


class _DummyPassage:
    def __init__(self, text: str) -> None:
        self.long_text = text


class _DummyRM:
    def __init__(
        self,
        data_source_id: str | None = None,
        provider: str | None = None,
        should_fail: bool = False,
    ) -> None:
        self.data_source_id = data_source_id
        self.provider = provider
        self.should_fail = should_fail

    def __call__(self, query: str, k: int = 3, **kwargs: Any) -> list[Any]:
        if self.should_fail:
            raise RuntimeError("RM retrieval failure")
        return [_DummyPassage(f"Passage {i} for {query}") for i in range(k)]


def test_sync_retrieve_execution(
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
        res = retrieve("What is OpenTelemetry?")
        assert len(res.passages) == 3

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "retrieval"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert attrs.get(GenAI.GEN_AI_REQUEST_TOP_K) == 3
    assert isinstance(attrs.get(GenAI.GEN_AI_REQUEST_TOP_K), int)
    assert (
        attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "What is OpenTelemetry?"
    )

    docs_attr = attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)
    assert isinstance(docs_attr, str)
    docs = json.loads(docs_attr)
    assert len(docs) == 3
    assert docs[0] == {"content": "Passage 0 for What is OpenTelemetry?"}
    assert docs[1] == {"content": "Passage 1 for What is OpenTelemetry?"}
    assert docs[2] == {"content": "Passage 2 for What is OpenTelemetry?"}


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
    assert attrs.get(GenAI.GEN_AI_REQUEST_TOP_K) == 2
    assert isinstance(attrs.get(GenAI.GEN_AI_REQUEST_TOP_K), int)
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
    assert attrs.get(GenAI.GEN_AI_REQUEST_TOP_K) == 4
    assert isinstance(attrs.get(GenAI.GEN_AI_REQUEST_TOP_K), int)
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
    assert attrs.get(GenAI.GEN_AI_REQUEST_TOP_K) == 2
    assert isinstance(attrs.get(GenAI.GEN_AI_REQUEST_TOP_K), int)
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


def test_retrieve_data_source_and_provider(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM(data_source_id="wiki_collection", provider="weaviate")
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
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == "weaviate"


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


@pytest.mark.skipif(
    not hasattr(dspy.Retrieve, "aforward"),
    reason="dspy.Retrieve.aforward not available in this DSPy version",
)
@pytest.mark.anyio
async def test_async_retrieve_execution(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    rm = _DummyRM()

    with (
        dspy.context(rm=rm),
        instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ),
    ):
        retrieve = dspy.Retrieve(k=2)
        res = await retrieve.aforward("What is OpenTelemetry?")
        assert len(res.passages) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "retrieval"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"


@pytest.mark.anyio
async def test_async_retrieve_aforward(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def dummy_aforward(
        self: Any, query: str, k: int | None = None, **kwargs: Any
    ) -> Any:
        k = k if k is not None else self.k
        return dspy.Prediction(
            passages=[f"Async passage {i} for {query}" for i in range(k)]
        )

    if not hasattr(dspy.Retrieve, "aforward"):
        monkeypatch.setattr(
            dspy.Retrieve, "aforward", dummy_aforward, raising=False
        )

    rm = _DummyRM()

    with (
        dspy.context(rm=rm),
        instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ),
    ):
        retrieve = dspy.Retrieve(k=3)
        res = await retrieve.aforward("What is OpenTelemetry?", k=2)
        assert len(res.passages) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "retrieval"
    assert span.status.status_code == StatusCode.UNSET
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
    assert attrs.get(GenAI.GEN_AI_REQUEST_TOP_K) == 2
    assert isinstance(attrs.get(GenAI.GEN_AI_REQUEST_TOP_K), int)
    assert (
        attrs.get(GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "What is OpenTelemetry?"
    )

    docs_attr = attrs.get(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS)
    assert isinstance(docs_attr, str)
    docs = json.loads(docs_attr)
    assert len(docs) == 2
    assert docs[0] == {"content": "Async passage 0 for What is OpenTelemetry?"}
    assert docs[1] == {"content": "Async passage 1 for What is OpenTelemetry?"}


@pytest.mark.anyio
async def test_async_retrieve_aforward_error(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_aforward(
        self: Any, query: str, k: int | None = None, **kwargs: Any
    ) -> Any:
        raise RuntimeError("Async RM retrieval failure")

    if not hasattr(dspy.Retrieve, "aforward"):
        monkeypatch.setattr(
            dspy.Retrieve, "aforward", failing_aforward, raising=False
        )

    rm = _DummyRM()

    with (
        dspy.context(rm=rm),
        instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ),
    ):
        retrieve = dspy.Retrieve(k=2)
        with pytest.raises(RuntimeError, match="Async RM retrieval failure"):
            await retrieve.aforward("Failing query")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "retrieval"
