# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding instrumentation tests.

Each test drives the library's real ``SentenceTransformer.encode`` over the
in-memory model built in ``test_utils``.
"""

import pytest
from sentence_transformers import SentenceTransformer

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .test_utils import MODEL_NAME, sentence_transformer

_EMBEDDINGS = GenAI.GenAiOperationNameValues.EMBEDDINGS.value


def test_encode_records_an_embeddings_span(
    instrument_no_content, span_exporter
):
    embeddings = sentence_transformer(dimension=8).encode(
        ["hello world", "embedding spans"]
    )
    assert embeddings.shape == (2, 8)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"embeddings {MODEL_NAME}"
    assert span.status.status_code == StatusCode.UNSET
    assert dict(span.attributes) == {
        GenAI.GEN_AI_OPERATION_NAME: _EMBEDDINGS,
        GenAI.GEN_AI_PROVIDER_NAME: "huggingface",
        GenAI.GEN_AI_REQUEST_MODEL: MODEL_NAME,
        GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT: 8,
    }


def test_encode_records_the_operation_duration_metric(
    instrument_no_content, metric_reader
):
    sentence_transformer().encode("hello")

    metrics = {
        metric.name: metric
        for resource_metric in metric_reader.get_metrics_data().resource_metrics
        for scope_metric in resource_metric.scope_metrics
        for metric in scope_metric.metrics
    }
    duration = metrics[gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION]
    assert [dict(point.attributes) for point in duration.data.data_points] == [
        {
            GenAI.GEN_AI_OPERATION_NAME: _EMBEDDINGS,
            GenAI.GEN_AI_PROVIDER_NAME: "huggingface",
            GenAI.GEN_AI_REQUEST_MODEL: MODEL_NAME,
        }
    ]
    # A local model returns no usage data, so no token usage is reported.
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE not in metrics


@pytest.mark.skipif(
    not hasattr(SentenceTransformer, "encode_query"),
    reason="encode_query and encode_document were added in 5.0",
)
def test_encode_query_and_document_record_one_span_each(
    instrument_no_content, span_exporter
):
    # encode_query and encode_document delegate to encode, the only wrapped
    # method, so each call records exactly one span.
    model = sentence_transformer()

    model.encode_query("hello")
    model.encode_document(["world"])

    spans = span_exporter.get_finished_spans()
    assert [s.attributes[GenAI.GEN_AI_OPERATION_NAME] for s in spans] == [
        _EMBEDDINGS,
        _EMBEDDINGS,
    ]


def test_unknown_model_name_falls_back_to_the_operation_name(
    instrument_no_content, span_exporter
):
    sentence_transformer(model_name=None).encode("hello")

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "embeddings"
    assert GenAI.GEN_AI_REQUEST_MODEL not in span.attributes


def test_runtime_error_is_recorded_and_reraised(
    instrument_no_content, span_exporter
):
    model = sentence_transformer(error=RuntimeError("out of memory"))

    with pytest.raises(RuntimeError, match="out of memory"):
        model.encode("hello")

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
