# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the duration and token usage metrics."""

from __future__ import annotations

from typing import Any

import ollama
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)

from .conftest import CHAT_MODEL, EMBEDDING_MODEL, HOST

MESSAGES = [{"role": "user", "content": "Say hello in three words."}]


def _metrics(metric_reader) -> dict[str, Any]:
    return {
        metric.name: metric
        for resource_metrics in metric_reader.get_metrics_data().resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }


def _token_usage(metric) -> dict[str, Any]:
    return {
        point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point
        for point in metric.data.data_points
    }


def test_chat_records_duration_and_token_usage(
    instrument_no_content, metric_reader, vcr
):
    with vcr.use_cassette("test_chat_records_a_chat_span.yaml"):
        response = ollama.Client(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES
        )

    metrics = _metrics(metric_reader)
    (duration,) = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ].data.data_points
    assert duration.count == 1
    assert duration.sum > 0
    assert duration.attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert duration.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "ollama"
    assert duration.attributes[GenAI.GEN_AI_REQUEST_MODEL] == CHAT_MODEL
    assert duration.attributes[GenAI.GEN_AI_RESPONSE_MODEL] == response.model
    assert duration.attributes[server_attributes.SERVER_ADDRESS] == "localhost"
    assert duration.attributes[server_attributes.SERVER_PORT] == 11434
    assert error_attributes.ERROR_TYPE not in duration.attributes

    usage = _token_usage(metrics[gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE])
    assert usage.keys() == {"input", "output"}
    assert usage["input"].sum == response.prompt_eval_count
    assert usage["output"].sum == response.eval_count


def test_chat_stream_records_token_usage_from_the_final_chunk(
    instrument_no_content, metric_reader, vcr
):
    with vcr.use_cassette(
        "test_chat_stream_records_the_drained_response.yaml"
    ):
        chunks = list(
            ollama.Client(host=HOST).chat(
                model=CHAT_MODEL, messages=MESSAGES, stream=True
            )
        )

    metrics = _metrics(metric_reader)
    (duration,) = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ].data.data_points
    assert duration.count == 1
    usage = _token_usage(metrics[gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE])
    assert usage["input"].sum == chunks[-1].prompt_eval_count
    assert usage["output"].sum == chunks[-1].eval_count


def test_chat_error_records_duration_with_error_type(
    instrument_no_content, metric_reader, vcr
):
    with vcr.use_cassette("test_chat_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            ollama.Client(host=HOST).chat(
                model="no-such-model:1b", messages=MESSAGES
            )

    metrics = _metrics(metric_reader)
    (duration,) = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ].data.data_points
    assert duration.count == 1
    assert (
        duration.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE not in metrics


def test_embed_records_duration_and_input_token_usage(
    instrument_no_content, metric_reader, vcr
):
    with vcr.use_cassette("test_embed_records_an_embeddings_span.yaml"):
        response = ollama.Client(host=HOST).embed(
            model=EMBEDDING_MODEL, input=["OpenTelemetry", "Ollama embeddings"]
        )

    metrics = _metrics(metric_reader)
    (duration,) = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ].data.data_points
    assert duration.count == 1
    assert duration.attributes[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"
    assert error_attributes.ERROR_TYPE not in duration.attributes

    usage = _token_usage(metrics[gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE])
    assert usage.keys() == {"input"}
    assert usage["input"].sum == response.prompt_eval_count


def test_embed_error_records_duration_with_error_type(
    instrument_no_content, metric_reader, vcr
):
    with vcr.use_cassette("test_embed_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            ollama.Client(host=HOST).embed(
                model="no-such-model:1b",
                input=["OpenTelemetry", "Ollama embeddings"],
            )

    metrics = _metrics(metric_reader)
    (duration,) = metrics[
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION
    ].data.data_points
    assert (
        duration.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )
