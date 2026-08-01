# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from groq import AsyncGroq, Groq

from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GenAiOperationNameValues,
)
from opentelemetry.trace import StatusCode


@pytest.mark.vcr
def test_chat_completions_basic(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_basic.yaml"):
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "user", "content": "Tell me a joke"},
            ],
        )

    assert response is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GEN_AI_SYSTEM) == "groq"
        or span.attributes.get("gen_ai.provider.name") == "groq"
    )
    assert (
        span.attributes.get(GEN_AI_OPERATION_NAME)
        == GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GEN_AI_REQUEST_MODEL] == "llama3-8b-8192"
    assert GEN_AI_RESPONSE_MODEL in span.attributes


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_completions_basic(
    instrument_no_content,
    span_exporter,
    vcr,
    async_groq_client: AsyncGroq,
):
    with vcr.use_cassette("test_async_chat_completions_basic.yaml"):
        response = await async_groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Tell me a joke"}],
        )

    assert response is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(GEN_AI_SYSTEM) == "groq"
        or span.attributes.get("gen_ai.provider.name") == "groq"
    )
    assert (
        span.attributes.get(GEN_AI_OPERATION_NAME)
        == GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GEN_AI_REQUEST_MODEL] == "llama3-8b-8192"


@pytest.mark.vcr
def test_chat_completions_streaming(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_streaming.yaml"):
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            stream=True,
        )
        for _ in response:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert (
        span.attributes.get(GEN_AI_SYSTEM) == "groq"
        or span.attributes.get("gen_ai.provider.name") == "groq"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_async_chat_completions_streaming(
    instrument_no_content,
    span_exporter,
    vcr,
    async_groq_client: AsyncGroq,
):
    with vcr.use_cassette("test_async_chat_completions_streaming.yaml"):
        response = await async_groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            stream=True,
        )
        async for _ in response:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1


@pytest.mark.vcr
def test_chat_completions_provider_error(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_provider_error.yaml"):
        with pytest.raises(Exception):
            groq_client.chat.completions.create(
                model="non-existent-model",
                messages=[{"role": "user", "content": "Tell me a joke"}],
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert "error.type" in span.attributes


@pytest.mark.vcr
def test_chat_completions_caller_side_error_inside_stream(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_caller_side_error.yaml"):
        with pytest.raises(ValueError, match="Caller error"):
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": "Tell me a long joke"}],
                stream=True,
            )
            with response:
                raise ValueError("Caller error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ValueError"


@pytest.mark.vcr
def test_chat_completions_stream_side_error_mid_iteration(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_stream_side_error.yaml"):
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "Tell me a long joke"}],
            stream=True,
        )
        from unittest import mock
        
        iterator_mock = mock.MagicMock()
        iterator_mock.__next__.side_effect = ConnectionError("Stream dropped")
        response._self_iterator = iterator_mock
        
        with pytest.raises(ConnectionError, match="Stream dropped"):
            for _ in response:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get("error.type") == "ConnectionError"
