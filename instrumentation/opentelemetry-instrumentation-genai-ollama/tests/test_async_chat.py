# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``AsyncClient.chat``."""

from __future__ import annotations

import asyncio
import json

import ollama
import pytest

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.trace import StatusCode

from .conftest import CHAT_MODEL, HOST

MESSAGES = [{"role": "user", "content": "Say hello in three words."}]


@pytest.mark.asyncio
async def test_async_chat_records_a_chat_span(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette("test_async_chat_records_a_chat_span.yaml"):
        response = await ollama.AsyncClient(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES
        )

    (span,) = span_exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == response.eval_count
    (output,) = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert output["parts"] == [
        {"type": "text", "content": response.message.content}
    ]


@pytest.mark.asyncio
async def test_async_chat_stream_records_the_drained_response(
    instrument_with_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_async_chat_stream_records_the_drained_response.yaml"
    ):
        stream = await ollama.AsyncClient(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES, stream=True
        )
        chunks = [chunk async for chunk in stream]

    (span,) = span_exporter.get_finished_spans()
    attrs = span.attributes
    assert attrs[GenAI.GEN_AI_REQUEST_STREAM] is True
    assert attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == chunks[-1].eval_count
    (output,) = json.loads(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])
    assert output["parts"] == [
        {
            "type": "text",
            "content": "".join(chunk.message.content for chunk in chunks),
        }
    ]


@pytest.mark.asyncio
async def test_concurrent_async_chats_record_their_own_usage(
    instrument_no_content, span_exporter, vcr
):
    prompts = ["Say one.", "Say two words now.", "Count to three please."]
    with vcr.use_cassette(
        "test_concurrent_async_chats_record_their_own_usage.yaml"
    ):
        client = ollama.AsyncClient(host=HOST)
        responses = await asyncio.gather(
            *(
                client.chat(
                    model=CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                for prompt in prompts
            )
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 3
    assert sorted(
        s.attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] for s in spans
    ) == sorted(response.eval_count for response in responses)
    assert len({s.context.trace_id for s in spans}) == 3


@pytest.mark.asyncio
async def test_async_chat_error_is_recorded_and_reraised(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_error_is_recorded_and_reraised.yaml"):
        with pytest.raises(ollama.ResponseError):
            await ollama.AsyncClient(host=HOST).chat(
                model="no-such-model:1b", messages=MESSAGES
            )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


@pytest.mark.asyncio
async def test_async_chat_stream_error_is_recorded(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette("test_chat_stream_error_is_recorded.yaml"):
        stream = await ollama.AsyncClient(host=HOST).chat(
            model="no-such-model:1b", messages=MESSAGES, stream=True
        )
        with pytest.raises(ollama.ResponseError):
            [chunk async for chunk in stream]

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[error_attributes.ERROR_TYPE]
        == "ollama._types.ResponseError"
    )


@pytest.mark.asyncio
async def test_async_chat_stream_closed_after_caller_error_ends_span(
    instrument_no_content, span_exporter, vcr
):
    with vcr.use_cassette(
        "test_async_chat_stream_records_the_drained_response.yaml"
    ):
        stream = await ollama.AsyncClient(host=HOST).chat(
            model=CHAT_MODEL, messages=MESSAGES, stream=True
        )
        with pytest.raises(ValueError):
            async for _ in stream:
                raise ValueError("boom")
        assert span_exporter.get_finished_spans() == ()
        await stream.aclose()

    # An error raised in the loop body never reaches the stream, so closing
    # it ends the span like any other early close.
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
