# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for DSPy Embedder instrumentation."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import dspy
import dspy.clients.embedding
import pytest

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode


def _custom_embed_fn(texts: list[str], **kwargs: Any) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_sync_embedder_custom_callable_batch(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ):
        embedder = dspy.Embedder(_custom_embed_fn, caching=False)
        result = embedder(["hello", "world"])
        assert result.shape == (2, 4)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings _custom_embed_fn"
    assert span.status.status_code == StatusCode.UNSET

    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "embeddings"
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == "dspy"
    assert attrs.get(GenAI.GEN_AI_REQUEST_MODEL) == "_custom_embed_fn"
    assert attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 4
    assert isinstance(attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT), int)

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metric_names = [
        m.name
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]
    assert "gen_ai.client.operation.duration" in metric_names


def test_sync_embedder_single_string_input(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(_custom_embed_fn, caching=False)
        result = embedder("hello")
        assert result.shape == (4,)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 4


@pytest.mark.anyio
async def test_async_embedder_custom_callable(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    class _EncoderModel:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

        def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.5, 0.6, 0.7] for _ in texts]

    encoder = _EncoderModel()

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(encoder.encode, caching=False)
        result = await embedder.acall(["first", "second"])
        assert result.shape == (2, 3)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings sentence-transformers/all-MiniLM-L6-v2"
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "embeddings"
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == "dspy"
    assert (
        attrs.get(GenAI.GEN_AI_REQUEST_MODEL)
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 3


def test_sync_embedder_hosted_model(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_embedding(
        model: str, input: list[str], **kwargs: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model="text-embedding-3-small-2024",
            data=[{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]} for _ in input],
            usage=SimpleNamespace(prompt_tokens=len(input) * 3),
        )

    fake_litellm = SimpleNamespace(
        cache=None,
        embedding=MagicMock(side_effect=_fake_embedding),
    )
    monkeypatch.setattr(
        dspy.clients.embedding,
        "_get_litellm",
        lambda: fake_litellm,
        raising=False,
    )
    monkeypatch.setattr(
        dspy.clients.embedding,
        "litellm",
        fake_litellm,
        raising=False,
    )

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(
            "openai/text-embedding-3-small",
            batch_size=2,
            caching=False,
            dimensions=5,
            encoding_format="float",
            api_base="https://api.example.com:8443/v1",
        )
        result = embedder(["one", "two", "three"])
        assert result.shape == (3, 5)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-3-small"
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_OPERATION_NAME) == "embeddings"
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == "openai"
    assert attrs.get(GenAI.GEN_AI_REQUEST_MODEL) == "text-embedding-3-small"
    assert GenAI.GEN_AI_RESPONSE_MODEL not in attrs
    assert GenAI.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 5
    assert attrs.get(GenAI.GEN_AI_REQUEST_ENCODING_FORMATS) == ("float",)
    assert attrs.get(server_attributes.SERVER_ADDRESS) == "api.example.com"
    assert attrs.get(server_attributes.SERVER_PORT) == 8443

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metric_names = [
        m.name
        for rm in metrics_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    ]
    assert "gen_ai.client.operation.duration" in metric_names


@pytest.mark.anyio
async def test_async_embedder_hosted_model(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_aembedding(
        model: str, input: list[str], **kwargs: Any
    ) -> SimpleNamespace:
        return SimpleNamespace(
            model="embed-english-v3.0",
            data=[{"embedding": [0.1, 0.2]} for _ in input],
            usage={"prompt_tokens": 7},
        )

    fake_litellm = SimpleNamespace(
        cache=None,
        aembedding=AsyncMock(side_effect=_fake_aembedding),
    )
    monkeypatch.setattr(
        dspy.clients.embedding,
        "_get_litellm",
        lambda: fake_litellm,
        raising=False,
    )
    monkeypatch.setattr(
        dspy.clients.embedding,
        "litellm",
        fake_litellm,
        raising=False,
    )

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(
            "cohere/embed-english-v3.0",
            caching=False,
        )
        result = await embedder.acall(["async text"])
        assert result.shape == (1, 2)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings embed-english-v3.0"
    attrs = span.attributes or {}
    assert attrs.get(GenAI.GEN_AI_PROVIDER_NAME) == "cohere"
    assert attrs.get(GenAI.GEN_AI_REQUEST_MODEL) == "embed-english-v3.0"
    assert GenAI.GEN_AI_RESPONSE_MODEL not in attrs
    assert GenAI.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT) == 2


def test_sync_embedder_error_records_failure(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    def _failing_embedder(texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service failed")

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(_failing_embedder, caching=False)
        with pytest.raises(RuntimeError, match="embedding service failed"):
            embedder(["hello"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "RuntimeError"


@pytest.mark.anyio
async def test_async_embedder_error_records_failure(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    def _failing_embedder(texts: list[str]) -> list[list[float]]:
        raise ConnectionError("async embedding network error")

    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(_failing_embedder, caching=False)
        with pytest.raises(
            ConnectionError, match="async embedding network error"
        ):
            await embedder.acall(["hello"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    attrs = span.attributes or {}
    assert attrs.get(error_attributes.ERROR_TYPE) == "ConnectionError"


def test_embedder_copy_and_deepcopy(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(_custom_embed_fn, caching=False)
        copied = copy.copy(embedder)
        deep_copied = copy.deepcopy(embedder)
        assert copied("hello").shape == (4,)
        assert deep_copied("world").shape == (4,)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 2


def test_embedder_constructor_kwargs_on_instance(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        DSPyInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        embedder = dspy.Embedder(_custom_embed_fn, caching=False)
        embedder.kwargs = {
            "api_base": "https://init.example.com:8443",
            "encoding_format": "base64",
        }
        embedder("hello")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes or {}
    assert attrs.get(server_attributes.SERVER_ADDRESS) == "init.example.com"
    assert attrs.get(server_attributes.SERVER_PORT) == 8443
    assert attrs.get(GenAI.GEN_AI_REQUEST_ENCODING_FORMATS) == ("base64",)
