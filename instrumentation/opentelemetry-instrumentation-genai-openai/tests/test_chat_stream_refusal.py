# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator

import pytest
from openai.types.chat import ChatCompletionChunk

from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    AsyncChatStreamWrapper,
    ChatStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _chunk(
    delta: dict, finish_reason: str | None = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
    )


_REFUSAL_CHUNKS = [
    _chunk({"role": "assistant", "content": None, "refusal": ""}),
    _chunk({"refusal": "I'm sorry, "}),
    _chunk({"refusal": "I can't help with that."}),
    _chunk({}, finish_reason="stop"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_streamed_refusal_is_captured_in_output_messages(
    tracer_provider, span_exporter, instrument_with_content, asynchronous: bool
) -> None:
    invocation = TelemetryHandler(tracer_provider=tracer_provider).inference(
        "openai", request_model="gpt-4o"
    )

    def produce() -> Iterator[ChatCompletionChunk]:
        yield from _REFUSAL_CHUNKS

    async def aproduce() -> AsyncIterator[ChatCompletionChunk]:
        for chunk in _REFUSAL_CHUNKS:
            await asyncio.sleep(0)
            yield chunk

    if asynchronous:
        stream = AsyncChatStreamWrapper(aproduce(), invocation, True)
        assert len([chunk async for chunk in stream]) == len(_REFUSAL_CHUNKS)
    else:
        stream = ChatStreamWrapper(produce(), invocation, True)
        assert len(list(stream)) == len(_REFUSAL_CHUNKS)

    (span,) = span_exporter.get_finished_spans()
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [
                {
                    "type": "text",
                    "content": "I'm sorry, I can't help with that.",
                }
            ],
            "finish_reason": "stop",
            "name": None,
        }
    ]
