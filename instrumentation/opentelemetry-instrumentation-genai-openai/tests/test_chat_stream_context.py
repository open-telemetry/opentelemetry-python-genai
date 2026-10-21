# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import pytest
from openai.types.chat import ChatCompletionChunk

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    AsyncChatStreamWrapper,
    ChatStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["drain", "close", "error"])
async def test_chat_stream_activates_only_during_reads(
    tracer_provider, span_exporter, caplog, asynchronous: bool, ending: str
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("openai", request_model="test-model")
    error = ConnectionError("stream failed")
    chunk = ChatCompletionChunk(
        id="test",
        created=1,
        model="test-model",
        object="chat.completion.chunk",
        choices=[],
    )

    def produce() -> Iterator[ChatCompletionChunk]:
        assert (
            trace.get_current_span().get_span_context()
            == trace.get_current_span(invocation.context).get_span_context()
        )
        yield chunk
        assert (
            trace.get_current_span().get_span_context()
            == trace.get_current_span(invocation.context).get_span_context()
        )
        if ending == "error":
            raise error

    async def aproduce() -> AsyncIterator[ChatCompletionChunk]:
        for item in produce():
            await asyncio.sleep(0)
            yield item

    if asynchronous:
        stream = AsyncChatStreamWrapper(aproduce(), invocation, False)
    else:
        stream = ChatStreamWrapper(produce(), invocation, False)
    assert context.get_current() is before
    if asynchronous:
        assert await asyncio.create_task(anext(stream)) is chunk
    else:
        assert next(stream) is chunk
    assert context.get_current() is before
    assert span_exporter.get_finished_spans() == ()

    if ending == "close":
        if asynchronous:
            await stream.aclose()
        else:
            stream.close()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            if asynchronous:
                await anext(stream)
            else:
                next(stream)
        assert raised.value is error
    elif asynchronous:
        assert [item async for item in stream] == []
    else:
        assert list(stream) == []
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    assert not [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= 40
    ]
