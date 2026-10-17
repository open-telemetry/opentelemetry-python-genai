# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SentenceTransformersInstrumentor lifecycle."""

from sentence_transformers import SentenceTransformer

from opentelemetry.instrumentation.genai.sentence_transformers import (
    SentenceTransformersInstrumentor,
)

from .test_utils import sentence_transformer


def test_instrumentation_dependencies():
    instrumentor = SentenceTransformersInstrumentor()
    assert instrumentor.instrumentation_dependencies() == (
        "sentence-transformers >= 3.3.0, < 7",
    )


def test_instrument_wraps_and_uninstrument_restores_encode(
    tracer_provider, logger_provider, meter_provider
):
    original_encode = SentenceTransformer.encode
    instrumentor = SentenceTransformersInstrumentor()

    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        assert SentenceTransformer.encode is not original_encode
    finally:
        instrumentor.uninstrument()

    assert SentenceTransformer.encode is original_encode


def test_uninstrument_stops_span_creation(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    instrumentor = SentenceTransformersInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    try:
        sentence_transformer().encode("hello")
        assert len(span_exporter.get_finished_spans()) == 1
    finally:
        instrumentor.uninstrument()

    span_exporter.clear()
    sentence_transformer().encode("hello")
    assert span_exporter.get_finished_spans() == ()
