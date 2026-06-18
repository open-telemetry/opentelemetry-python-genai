# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the embeddings invocation builder and wrappers.

These tests replace the legacy ``test_request_attributes.py`` coverage
that targeted the removed ``get_llm_request_attributes`` /
``_get_embeddings_span_name`` helpers, and exercise the new
``EmbeddingInvocation``-based flow added by the util-genai migration.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from httpx import URL

from opentelemetry.instrumentation.genai.openai.patch import (
    _create_embedding_invocation as create_embedding_invocation,
)
from opentelemetry.instrumentation.genai.openai.patch import (
    embeddings_create,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _make_client(base_url: str | URL | None = "https://api.openai.com/v1"):
    """Return a stand-in for an OpenAI client with the attributes we read."""
    return SimpleNamespace(_client=SimpleNamespace(base_url=base_url))


@pytest.fixture(autouse=True)
def fixture_vcr():
    """No VCR needed for these unit tests."""
    yield


@pytest.fixture
def handler(tracer_provider, meter_provider, logger_provider):
    return TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )


# ─── create_embedding_invocation: request_model behavior ────────────────────


def test_model_omitted_when_missing(handler, span_exporter):
    """When 'model' is not in kwargs, GEN_AI_REQUEST_MODEL must be unset."""
    invocation = create_embedding_invocation(handler, {}, _make_client())
    invocation.stop()

    assert invocation.request_model is None

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Span name must NOT include a trailing space / empty model.
    assert span.name == "embeddings"
    assert GenAIAttributes.GEN_AI_REQUEST_MODEL not in span.attributes


def test_model_preserved_when_provided(handler, span_exporter):
    """When 'model' is in kwargs, GEN_AI_REQUEST_MODEL must be set."""
    invocation = create_embedding_invocation(
        handler, {"model": "text-embedding-3-small"}, _make_client()
    )
    invocation.stop()

    assert invocation.request_model == "text-embedding-3-small"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "text-embedding-3-small"
    )


# ─── create_embedding_invocation: server address / port ─────────────────────


def test_server_address_and_port_from_client(handler, span_exporter):
    client = _make_client(URL("http://localhost:8080/v1"))
    invocation = create_embedding_invocation(handler, {"model": "m"}, client)
    invocation.stop()

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes[ServerAttributes.SERVER_ADDRESS] == "localhost"
    assert span.attributes[ServerAttributes.SERVER_PORT] == 8080


# ─── create_embedding_invocation: dimensions / encoding_format ──────────────


def test_dimensions_propagated_to_metric_attributes(handler):
    """Request-side ``dimensions`` should be exposed as a metric attribute."""
    invocation = create_embedding_invocation(
        handler,
        {"model": "m", "dimensions": 256},
        _make_client(),
    )
    try:
        assert invocation.dimension_count == 256
        assert (
            invocation.metric_attributes[
                GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
            ]
            == 256
        )
        assert isinstance(
            invocation.metric_attributes[
                GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
            ],
            int,
        )
    finally:
        invocation.stop()


def test_dimensions_omitted_when_not_provided(handler):
    invocation = create_embedding_invocation(
        handler, {"model": "m"}, _make_client()
    )
    try:
        assert invocation.dimension_count is None
        assert (
            GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
            not in invocation.metric_attributes
        )
    finally:
        invocation.stop()


def test_encoding_format_mapped_to_invocation(handler):
    invocation = create_embedding_invocation(
        handler,
        {"model": "m", "encoding_format": "base64"},
        _make_client(),
    )
    try:
        assert invocation.encoding_formats == ["base64"]
    finally:
        invocation.stop()


def test_encoding_format_omitted_when_not_provided(handler):
    invocation = create_embedding_invocation(
        handler, {"model": "m"}, _make_client()
    )
    try:
        assert invocation.encoding_formats is None
    finally:
        invocation.stop()


# ─── _set_embeddings_response_properties via the wrapper ────────────────────


def _fake_embedding_response(
    *, model: str = "m", dim: int = 3, prompt_tokens: int = 7
):
    return SimpleNamespace(
        model=model,
        data=[SimpleNamespace(embedding=[0.0] * dim)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens),
    )


def test_response_derived_dimension_count_lands_on_metric_attributes(
    handler, span_exporter, metric_reader
):
    """When ``dimensions`` is inferred from the response, it must still be on metrics."""
    response = _fake_embedding_response(dim=8)

    def wrapped(*_args, **_kwargs):
        return response

    traced = embeddings_create(handler)
    # Note: no 'dimensions' in kwargs -> only the response carries it.
    result = traced(wrapped, _make_client(), (), {"model": "m"})
    assert result is response

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 8
    )

    metrics = metric_reader.get_metrics_data()
    found_dim_on_metric = False
    for resource_metric in metrics.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                for point in metric.data.data_points:
                    if (
                        point.attributes.get(
                            GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT
                        )
                        == 8
                    ):
                        found_dim_on_metric = True
    assert found_dim_on_metric, (
        "dimension count should be propagated to the metric attributes "
        "when derived from the response"
    )


def test_extraction_error_is_swallowed_and_does_not_break_wrapped_call(
    handler, span_exporter
):
    """If response-property extraction raises, the wrapper must not."""

    class _BadResponse:
        """Accessing ``.data`` raises - simulates an unexpected SDK shape."""

        model = "m"
        usage = SimpleNamespace(prompt_tokens=1)

        @property
        def data(self) -> Any:  # noqa: D401
            raise RuntimeError("unexpected SDK shape")

    bad_response = _BadResponse()

    def wrapped(*_args, **_kwargs):
        return bad_response

    traced = embeddings_create(handler)
    # Must not raise even though extraction blows up.
    result = traced(wrapped, _make_client(), (), {"model": "m"})
    assert result is bad_response

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # Span should be successfully stopped (no error attribute) because the
    # wrapped call itself succeeded.
    assert "error.type" not in span.attributes


def test_wrapped_call_exception_is_recorded_and_reraised(
    handler, span_exporter
):
    def wrapped(*_args, **_kwargs):
        raise ValueError("boom")

    traced = embeddings_create(handler)
    with pytest.raises(ValueError, match="boom"):
        traced(wrapped, _make_client(), (), {"model": "m"})

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes["error.type"] == "ValueError"
