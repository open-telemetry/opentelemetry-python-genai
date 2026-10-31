# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.bedrock.stream import (
    AsyncBedrockAgentEventStreamWrapper,
    AsyncBedrockConverseStreamWrapper,
    AsyncBedrockInvokeModelStreamWrapper,
    AsyncBedrockRetrieveAndGenerateStreamWrapper,
    AsyncBedrockStreamingBodyWrapper,
    BedrockAgentEventStreamWrapper,
    BedrockConverseStreamWrapper,
    BedrockInvokeModelStreamWrapper,
    BedrockRetrieveAndGenerateStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def _assert_invocation_is_current(invocation: Any) -> None:
    assert (
        trace.get_current_span().get_span_context()
        == trace.get_current_span(invocation.context).get_span_context()
    )


class _ContextCheckingAsyncBody:
    def __init__(self, data: bytes, invocation: Any) -> None:
        self._data = data
        self._invocation = invocation
        self.closed = False

    async def read(self, _amt: int | None = None) -> bytes:
        _assert_invocation_is_current(self._invocation)
        data, self._data = self._data, b""
        return data

    async def __aenter__(self) -> _ContextCheckingAsyncBody:
        _assert_invocation_is_current(self._invocation)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        _assert_invocation_is_current(self._invocation)
        self.closed = True

    async def aclose(self) -> None:
        _assert_invocation_is_current(self._invocation)
        self.closed = True

    def close(self) -> None:
        _assert_invocation_is_current(self._invocation)
        self.closed = True


STREAM_WRAPPERS = [
    pytest.param(BedrockConverseStreamWrapper, False, id="converse-sync"),
    pytest.param(AsyncBedrockConverseStreamWrapper, True, id="converse-async"),
    pytest.param(
        BedrockInvokeModelStreamWrapper, False, id="invoke-model-sync"
    ),
    pytest.param(
        AsyncBedrockInvokeModelStreamWrapper,
        True,
        id="invoke-model-async",
    ),
    pytest.param(BedrockAgentEventStreamWrapper, False, id="agent-sync"),
    pytest.param(AsyncBedrockAgentEventStreamWrapper, True, id="agent-async"),
    pytest.param(
        BedrockRetrieveAndGenerateStreamWrapper,
        False,
        id="retrieve-and-generate-sync",
    ),
    pytest.param(
        AsyncBedrockRetrieveAndGenerateStreamWrapper,
        True,
        id="retrieve-and-generate-async",
    ),
]


_AGENT_WRAPPER_TYPES = (
    BedrockAgentEventStreamWrapper,
    AsyncBedrockAgentEventStreamWrapper,
)


def _invocation_for(tracer_provider: Any, wrapper_cls: type) -> Any:
    """Start the invocation the way patch.py does for the given wrapper type.

    patch.py starts agent streams with ``handler.invoke_remote_agent()``, so those
    wrappers receive a ``RemoteAgentInvocation``. Every other streaming wrapper in
    this parametrisation is an inference stream started with
    ``handler.inference()``.
    """
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    if wrapper_cls in _AGENT_WRAPPER_TYPES:
        return handler.invoke_remote_agent(
            provider="aws.bedrock",
            server_address="bedrock-agent-runtime.us-east-1.amazonaws.com",
            server_port=443,
        )
    return handler.inference("bedrock", request_model="test-model")


@pytest.mark.asyncio
@pytest.mark.parametrize(("wrapper_cls", "asynchronous"), STREAM_WRAPPERS)
@pytest.mark.parametrize("ending", ["drain", "close", "error"])
async def test_stream_activates_invocation_only_during_reads(
    tracer_provider,
    span_exporter,
    caplog,
    wrapper_cls: type,
    asynchronous: bool,
    ending: str,
) -> None:
    before = context.get_current()
    invocation = _invocation_for(tracer_provider, wrapper_cls)
    error = ConnectionError("stream failed")
    chunk: dict[str, Any] = {}

    def produce() -> Iterator[dict[str, Any]]:
        _assert_invocation_is_current(invocation)
        yield chunk
        _assert_invocation_is_current(invocation)
        if ending == "error":
            raise error

    async def aproduce() -> AsyncIterator[dict[str, Any]]:
        for item in produce():
            await asyncio.sleep(0)
            yield item

    source = aproduce() if asynchronous else produce()
    stream = wrapper_cls(source, invocation)

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
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    if ending == "error":
        assert spans[0].status.status_code == StatusCode.ERROR
        assert spans[0].attributes is not None
        assert (
            spans[0].attributes[ErrorAttributes.ERROR_TYPE]
            == "ConnectionError"
        )
    assert not [
        record
        for record in caplog.records
        if record.name == "opentelemetry.context" and record.levelno >= 40
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("wrapper_cls", "asynchronous"), STREAM_WRAPPERS)
async def test_stream_context_manager_records_caller_error(
    tracer_provider,
    span_exporter,
    wrapper_cls: type,
    asynchronous: bool,
) -> None:
    before = context.get_current()
    invocation = _invocation_for(tracer_provider, wrapper_cls)
    error = ConnectionError("caller failed")
    chunks: tuple[dict[str, Any], ...] = ({}, {})

    async def aproduce() -> AsyncIterator[dict[str, Any]]:
        for chunk in chunks:
            await asyncio.sleep(0)
            yield chunk

    source = aproduce() if asynchronous else iter(chunks)
    stream = wrapper_cls(source, invocation)

    with pytest.raises(ConnectionError) as raised:
        if asynchronous:
            async with stream:
                assert await anext(stream) is chunks[0]
                assert context.get_current() is before
                raise error
        else:
            with stream:
                assert next(stream) is chunks[0]
                assert context.get_current() is before
                raise error

    assert raised.value is error
    assert context.get_current() is before
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes is not None
    assert spans[0].attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio
async def test_streaming_body_activates_invocation_only_while_reading(
    tracer_provider,
    span_exporter,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("bedrock", request_model="test-model")
    body = _ContextCheckingAsyncBody(b"{}", invocation)
    stream = AsyncBedrockStreamingBodyWrapper(
        body,
        invocation,
        {"body": body},
        capture_content=False,
    )

    assert context.get_current() is before
    assert await asyncio.create_task(stream.read()) == b"{}"
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["aclose", "close"])
async def test_streaming_body_scopes_early_close(
    tracer_provider,
    span_exporter,
    method: str,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("bedrock", request_model="test-model")
    body = _ContextCheckingAsyncBody(b"{}", invocation)
    stream = AsyncBedrockStreamingBodyWrapper(
        body,
        invocation,
        {"body": body},
        capture_content=False,
    )

    assert context.get_current() is before
    if method == "aclose":
        await stream.aclose()
    else:
        stream.close()
    assert context.get_current() is before
    assert body.closed
    assert len(span_exporter.get_finished_spans()) == 1


@pytest.mark.asyncio
async def test_streaming_body_scopes_caller_error(
    tracer_provider,
    span_exporter,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("bedrock", request_model="test-model")
    body = _ContextCheckingAsyncBody(b"{}", invocation)
    stream = AsyncBedrockStreamingBodyWrapper(
        body,
        invocation,
        {"body": body},
        capture_content=False,
    )
    error = RuntimeError("caller failed")

    with pytest.raises(RuntimeError) as raised:
        async with stream:
            assert context.get_current() is before
            raise error
    assert raised.value is error
    assert context.get_current() is before
    assert body.closed
    assert len(span_exporter.get_finished_spans()) == 1


def test_streaming_body_scopes_abandonment(
    tracer_provider,
    span_exporter,
    caplog,
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider,
    ).inference("bedrock", request_model="test-model")
    body = _ContextCheckingAsyncBody(b"{}", invocation)
    stream = AsyncBedrockStreamingBodyWrapper(
        body,
        invocation,
        {"body": body},
        capture_content=False,
    )

    assert context.get_current() is before
    del stream
    gc.collect()

    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    assert not [
        record
        for record in caplog.records
        if record.name == "opentelemetry.context" and record.levelno >= 40
    ]
