# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``embed`` on the sync and async clients and the module."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any

import ollama
import pytest

from opentelemetry import trace
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.trace import StatusCode

from .conftest import EMBEDDING_MODEL, HOST

INPUT = ["OpenTelemetry", "Ollama embeddings"]


def _assert_embeddings_span(span, response) -> None:
    attrs = span.attributes
    assert span.name == f"embeddings {EMBEDDING_MODEL}"
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "ollama"
    assert attrs[GenAI.GEN_AI_REQUEST_MODEL] == EMBEDDING_MODEL
    assert attrs[GenAI.GEN_AI_RESPONSE_MODEL] == response.model
    # Only a requested size is recorded, not the model's default one.
    assert GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT not in attrs
    assert attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == response.prompt_eval_count
    assert attrs[server_attributes.SERVER_ADDRESS] == "localhost"


def test_embed_records_an_embeddings_span(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_embed_records_an_embeddings_span.yaml"):
        response = ollama.Client(host=HOST).embed(
            model=EMBEDDING_MODEL, input=INPUT
        )

    (span,) = span_exporter.get_finished_spans()
    _assert_embeddings_span(span, response)


@pytest.mark.asyncio
async def test_async_embed_records_an_embeddings_span(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_async_embed_records_an_embeddings_span.yaml"):
        response = await ollama.AsyncClient(host=HOST).embed(
            model=EMBEDDING_MODEL, input=INPUT
        )

    (span,) = span_exporter.get_finished_spans()
    _assert_embeddings_span(span, response)


def test_module_level_embed_is_instrumented(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_module_level_embed_is_instrumented.yaml"):
        ollama.embed(model=EMBEDDING_MODEL, input="OpenTelemetry")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"


@pytest.mark.skipif(
    "dimensions" not in inspect.signature(ollama.Client.embed).parameters,
    reason="the dimensions parameter is not available on this ollama version",
)
def test_embed_records_requested_dimensions(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_embed_records_requested_dimensions.yaml"):
        response = ollama.Client(host=HOST).embed(
            model=EMBEDDING_MODEL, input=INPUT, dimensions=256
        )

    assert len(response.embeddings[0]) == 256
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 256


def test_embed_error_is_recorded_and_reraised(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_embed_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            ollama.Client(host=HOST).embed(
                model="no-such-model:1b", input=INPUT
            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


@pytest.mark.asyncio
async def test_async_embed_error_is_recorded_and_reraised(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_embed_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            await ollama.AsyncClient(host=HOST).embed(
                model="no-such-model:1b", input=INPUT
            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


@pytest.mark.parametrize(
    "value, cassette",
    [
        ("", "test_embed_with_empty_string_records_no_span.yaml"),
        ([], "test_embed_with_empty_list_records_no_span.yaml"),
    ],
    ids=["empty-string", "empty-list"],
)
def test_embed_without_input_records_no_span(
    instrument_no_content, span_exporter, vcr, value, cassette
):
    # Ollama embeds nothing for an empty input.
    with vcr.use_cassette(cassette):
        response = ollama.Client(host=HOST).embed(
            model=EMBEDDING_MODEL, input=value
        )

    assert response.embeddings == []
    assert span_exporter.get_finished_spans() == ()


def _embed_cases(success_cassette: str) -> list[Any]:
    return [
        pytest.param(EMBEDDING_MODEL, success_cassette, id="success"),
        pytest.param(
            "no-such-model:1b",
            "test_embed_error_is_recorded_and_reraised.yaml",
            id="error",
        ),
    ]


def _raises(model: str) -> Any:
    if model == EMBEDDING_MODEL:
        return nullcontext()
    return pytest.raises(ollama.ResponseError)


@pytest.mark.parametrize(
    "model, cassette",
    _embed_cases("test_embed_records_an_embeddings_span.yaml"),
)
def test_embed_span_is_current_only_during_the_request(
    instrument_no_content, span_exporter, vcr, model, cassette
):
    seen: list[int] = []

    def hook(_request: Any) -> None:
        seen.append(trace.get_current_span().get_span_context().span_id)

    with vcr.use_cassette(cassette), _raises(model):
        ollama.Client(host=HOST, event_hooks={"request": [hook]}).embed(
            model=model, input=INPUT
        )
    assert not trace.get_current_span().get_span_context().is_valid

    (span,) = span_exporter.get_finished_spans()
    assert seen == [span.context.span_id]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, cassette",
    _embed_cases("test_async_embed_records_an_embeddings_span.yaml"),
)
async def test_async_embed_span_is_current_only_during_the_request(
    instrument_no_content, span_exporter, vcr, model, cassette
):
    seen: list[int] = []

    async def hook(_request: Any) -> None:
        seen.append(trace.get_current_span().get_span_context().span_id)

    with vcr.use_cassette(cassette), _raises(model):
        await ollama.AsyncClient(
            host=HOST, event_hooks={"request": [hook]}
        ).embed(model=model, input=INPUT)
    assert not trace.get_current_span().get_span_context().is_valid

    (span,) = span_exporter.get_finished_spans()
    assert seen == [span.context.span_id]
