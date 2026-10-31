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
