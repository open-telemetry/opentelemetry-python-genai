# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``AsyncClient.chat``."""

from __future__ import annotations

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
