# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``embed`` on the sync and async clients and the module."""

from __future__ import annotations

import ollama
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import server_attributes

from .conftest import EMBEDDING_MODEL, HOST

INPUT = ["OpenTelemetry", "Ollama embeddings"]


def _assert_embeddings_span(span, response) -> None:
    attrs = span.attributes
    assert span.name == f"embeddings {EMBEDDING_MODEL}"
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "embeddings"
    assert attrs[GenAI.GEN_AI_PROVIDER_NAME] == "ollama"
    assert attrs[GenAI.GEN_AI_REQUEST_MODEL] == EMBEDDING_MODEL
    assert attrs[GenAI.GEN_AI_RESPONSE_MODEL] == response.model
    assert attrs[GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == len(
        response.embeddings[0]
    )
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
