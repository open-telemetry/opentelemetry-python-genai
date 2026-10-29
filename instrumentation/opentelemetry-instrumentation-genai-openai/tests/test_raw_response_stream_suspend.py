# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from openai import AsyncStream, Stream

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.openai._raw_response import (
    wrap_stream_result,
)
from opentelemetry.instrumentation.genai.openai.chat_wrappers import (
    AsyncChatStreamWrapper,
    ChatStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler

from .test_raw_response_proxy import _AsyncRawResponse, _RawResponse


class _EmptyStream(Stream):
    def __init__(self):
        pass

    def __iter__(self):
        return iter(())

    def close(self):
        pass


class _EmptyAsyncStream(AsyncStream):
    def __init__(self):
        pass

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def close(self):
        pass


def _start(tracer_provider):
    return TelemetryHandler(tracer_provider=tracer_provider).inference(
        "openai", request_model="m"
    )


def test_raw_response_span_not_current_after_create(tracer_provider):
    before = context.get_current()
    raw = wrap_stream_result(
        ChatStreamWrapper,
        _RawResponse(_EmptyStream()),
        _start(tracer_provider),
        False,
    )
    assert context.get_current() is before
    raw.http_response.close()


def test_unparsed_raw_response_stops_inside_its_span(tracer_provider):
    invocation = _start(tracer_provider)
    stop = invocation.stop
    current_at_stop = []

    def recording_stop():
        current_at_stop.append(trace.get_current_span())
        stop()

    invocation.stop = recording_stop
    raw = wrap_stream_result(
        ChatStreamWrapper, _RawResponse(_EmptyStream()), invocation, False
    )
    before = context.get_current()
    with tracer_provider.get_tracer("t").start_as_current_span("caller"):
        raw.http_response.close()

    assert current_at_stop == [invocation.span]
    assert context.get_current() is before


def test_raw_response_parse_under_caller_span_then_drain(
    tracer_provider, span_exporter
):
    before = context.get_current()
    raw = wrap_stream_result(
        ChatStreamWrapper,
        _RawResponse(_EmptyStream()),
        _start(tracer_provider),
        False,
    )
    with tracer_provider.get_tracer("t").start_as_current_span("caller"):
        stream = raw.parse()
    assert list(stream) == []
    assert len(span_exporter.get_finished_spans()) == 2
    assert context.get_current() is before


def test_async_raw_response_parse_under_caller_span_then_drain(
    tracer_provider, span_exporter
):
    async def run():
        before = context.get_current()
        raw = wrap_stream_result(
            AsyncChatStreamWrapper,
            _AsyncRawResponse(_EmptyAsyncStream()),
            _start(tracer_provider),
            False,
        )
        assert context.get_current() is before
        with tracer_provider.get_tracer("t").start_as_current_span("caller"):
            stream = await raw.parse()
        assert [chunk async for chunk in stream] == []
        assert len(span_exporter.get_finished_spans()) == 2
        assert context.get_current() is before

    asyncio.run(run())
