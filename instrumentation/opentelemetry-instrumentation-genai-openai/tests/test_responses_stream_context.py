# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from opentelemetry import context, trace
from opentelemetry.instrumentation.genai.openai.response_wrappers import (
    AsyncResponseStreamManagerWrapper,
    AsyncResponseStreamWrapper,
    ResponseStreamManagerWrapper,
    ResponseStreamWrapper,
)
from opentelemetry.util.genai.handler import TelemetryHandler


def _assert_no_detach_errors(caplog) -> None:
    assert not [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= 40
    ]


class _Manager:
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __enter__(self) -> Any:
        return self._stream

    def __exit__(self, *exc: Any) -> bool:
        return False

    async def __aenter__(self) -> Any:
        return self._stream

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def _events(
    invocation: Any, event: Any, error: BaseException | None
) -> Iterator[Any]:
    assert (
        trace.get_current_span().get_span_context()
        == trace.get_current_span(invocation.context).get_span_context()
    )
    yield event
    assert (
        trace.get_current_span().get_span_context()
        == trace.get_current_span(invocation.context).get_span_context()
    )
    if error is not None:
        raise error


async def _aevents(
    invocation: Any, event: Any, error: BaseException | None
) -> AsyncIterator[Any]:
    for item in _events(invocation, event, error):
        await asyncio.sleep(0)
        yield item


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("ending", ["drain", "close", "error"])
async def test_responses_stream_activates_only_during_reads(
    tracer_provider, span_exporter, caplog, asynchronous: bool, ending: str
) -> None:
    before = context.get_current()
    invocation = TelemetryHandler(tracer_provider=tracer_provider).inference(
        "openai", request_model="test-model"
    )
    error = ConnectionError("stream failed") if ending == "error" else None
    event = SimpleNamespace(type="response.output_text.delta")

    if asynchronous:
        stream = AsyncResponseStreamWrapper(
            _aevents(invocation, event, error), invocation, False
        )
    else:
        stream = ResponseStreamWrapper(
            _events(invocation, event, error), invocation, False
        )
    assert context.get_current() is before
    if asynchronous:
        assert await asyncio.create_task(anext(stream)) is event
    else:
        assert next(stream) is event
    assert context.get_current() is before
    assert span_exporter.get_finished_spans() == ()

    async def drain() -> list[Any]:
        return [item async for item in stream]

    if ending == "close":
        if asynchronous:
            await stream.aclose()
        else:
            stream.close()
    elif ending == "error":
        with pytest.raises(ConnectionError) as raised:
            if asynchronous:
                await asyncio.create_task(anext(stream))
            else:
                next(stream)
        assert raised.value is error
    elif asynchronous:
        assert await asyncio.create_task(drain()) == []
    else:
        assert list(stream) == []
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    _assert_no_detach_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_stream_manager_activates_only_during_reads(
    tracer_provider, span_exporter, caplog, asynchronous: bool
) -> None:
    before = context.get_current()
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocations: list[Any] = []
    event = SimpleNamespace(type="response.output_text.delta")

    def factory() -> Any:
        invocations.append(
            handler.inference("openai", request_model="test-model")
        )
        return invocations[-1]

    def produce() -> Any:
        return (
            _aevents(invocations[-1], event, None)
            if asynchronous
            else _events(invocations[-1], event, None)
        )

    class Manager(_Manager):
        def __init__(self) -> None:
            super().__init__(None)

        def __enter__(self) -> Any:
            return produce()

        async def __aenter__(self) -> Any:
            return produce()

    if asynchronous:
        manager = AsyncResponseStreamManagerWrapper(Manager(), factory, False)
        async with manager as stream:
            assert context.get_current() is before
            assert await asyncio.create_task(anext(stream)) is event
            assert context.get_current() is before
            assert span_exporter.get_finished_spans() == ()
    else:
        manager = ResponseStreamManagerWrapper(Manager(), factory, False)
        with manager as stream:
            assert context.get_current() is before
            assert next(stream) is event
            assert context.get_current() is before
            assert span_exporter.get_finished_spans() == ()
    assert context.get_current() is before
    assert len(span_exporter.get_finished_spans()) == 1
    _assert_no_detach_errors(caplog)


class _RecordingHook:
    def __init__(self) -> None:
        self.seen: list[Any] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.seen.append(trace.get_current_span().get_span_context())


class _Response:
    def __init__(self) -> None:
        self.closed_in: list[Any] = []

    def close(self) -> None:
        self.closed_in.append(trace.get_current_span().get_span_context())

    async def aclose(self) -> None:
        self.close()


class _Stream:
    def __init__(self, response: _Response) -> None:
        self._response = response

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __aiter__(self) -> _Stream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_stream_response_close_activates_invocation(
    tracer_provider, span_exporter, caplog, asynchronous: bool
) -> None:
    before = context.get_current()
    hook = _RecordingHook()
    invocation = TelemetryHandler(
        tracer_provider=tracer_provider, completion_hook=hook
    ).inference("openai", request_model="test-model")
    expected = trace.get_current_span(invocation.context).get_span_context()
    response = _Response()

    if asynchronous:
        stream = AsyncResponseStreamWrapper(
            _Stream(response), invocation, False
        )
    else:
        stream = ResponseStreamWrapper(_Stream(response), invocation, False)
    assert context.get_current() is before

    if asynchronous:
        await stream.response.aclose()
    else:
        stream.response.close()

    assert context.get_current() is before
    assert response.closed_in == [expected]
    assert hook.seen == [expected]
    assert len(span_exporter.get_finished_spans()) == 1
    _assert_no_detach_errors(caplog)
