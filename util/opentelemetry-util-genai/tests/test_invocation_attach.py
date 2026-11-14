# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import pytest

from opentelemetry import context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import get_current_span
from opentelemetry.util.genai._invocation import GenAIInvocation
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)

_FACTORIES: dict[str, Callable[..., GenAIInvocation]] = {
    "inference": lambda h, **kw: h.inference("prov", **kw),
    "embedding": lambda h, **kw: h.embedding("prov", **kw),
    "retrieval": lambda h, **kw: h.retrieval(**kw),
    "fetch_response": lambda h, **kw: h.fetch_response(
        "prov", response_id="r", **kw
    ),
    "tool": lambda h, **kw: h.tool("tool", **kw),
    "local_agent": lambda h, **kw: h.invoke_local_agent(agent_name="a", **kw),
    "remote_agent": lambda h, **kw: h.invoke_remote_agent("prov", **kw),
    "workflow": lambda h, **kw: h.workflow("flow", **kw),
}


@pytest.fixture(name="span_exporter")
def fixture_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(name="handler")
def fixture_handler(span_exporter: InMemorySpanExporter) -> TelemetryHandler:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return TelemetryHandler(tracer_provider=provider)


@pytest.fixture(name="caller_span")
def fixture_caller_span(handler: TelemetryHandler) -> Iterator[Any]:
    with handler._tracer.start_as_current_span("caller") as span:
        yield span


def _no_context_errors(caplog: pytest.LogCaptureFixture) -> bool:
    return not [
        r
        for r in caplog.records
        if r.name == "opentelemetry.context" and r.levelno >= logging.ERROR
    ]


@pytest.mark.parametrize("factory", _FACTORIES.values(), ids=_FACTORIES.keys())
def test_is_current_only_inside_with(
    handler: TelemetryHandler,
    caller_span: Any,
    factory: Callable[..., GenAIInvocation],
) -> None:
    invocation = factory(handler, _attach_to_context=False)
    assert get_current_span() is caller_span

    with invocation:
        assert get_current_span() is invocation.span

    assert get_current_span() is caller_span
    assert invocation._finished


@pytest.mark.parametrize("factory", _FACTORIES.values(), ids=_FACTORIES.keys())
def test_activate_then_stop(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    factory: Callable[..., GenAIInvocation],
) -> None:
    invocation = factory(handler, _attach_to_context=False)
    with invocation.activate():
        assert get_current_span() is invocation.span
        nested = handler.tool("nested")
        nested.stop()
    assert get_current_span() is caller_span
    invocation.stop()

    assert get_current_span() is caller_span
    by_id = {s.context.span_id: s for s in span_exporter.get_finished_spans()}
    outer = by_id[invocation.span.get_span_context().span_id]
    inner = by_id[nested.span.get_span_context().span_id]
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert outer.parent is not None
    assert outer.parent.span_id == caller_span.get_span_context().span_id
    assert _no_context_errors(caplog)


def test_with_records_error_and_restores_context(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    error = ValueError("boom")
    with pytest.raises(ValueError):
        with handler.inference("prov", _attach_to_context=False):
            raise error

    assert get_current_span() is caller_span
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ValueError"


def test_default_attaches_on_creation(
    handler: TelemetryHandler, caller_span: Any
) -> None:
    invocation = handler.inference("prov")
    assert get_current_span() is invocation.span
    invocation.stop()
    assert get_current_span() is caller_span


class _SyncWrapper(SyncStreamWrapper[str]):
    def __init__(self, stream: Any, invocation: GenAIInvocation) -> None:
        super().__init__(stream, invocation)
        self._self_genai_invocation = invocation
        self._self_seen: list[Any] = []

    def _process_chunk(self, chunk: str) -> None:
        self._self_seen.append(get_current_span())

    def _on_stream_end(self) -> None:
        self._self_genai_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_genai_invocation.fail(error)


class _AsyncWrapper(AsyncStreamWrapper[str]):
    def __init__(self, stream: Any, invocation: GenAIInvocation) -> None:
        super().__init__(stream, invocation)
        self._self_genai_invocation = invocation
        self._self_seen: list[Any] = []

    def _process_chunk(self, chunk: str) -> None:
        self._self_seen.append(get_current_span())

    def _on_stream_end(self) -> None:
        self._self_genai_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_genai_invocation.fail(error)


def test_sync_stream_scopes_unattached_invocation_to_reads(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    invocation = handler.inference("prov", _attach_to_context=False)

    def chunks() -> Iterator[str]:
        assert get_current_span() is invocation.span
        yield "a"
        assert get_current_span() is invocation.span
        yield "b"

    stream = _SyncWrapper(chunks(), invocation)
    assert get_current_span() is caller_span

    for _ in stream:
        assert get_current_span() is caller_span

    assert stream._self_seen == [invocation.span, invocation.span]
    assert get_current_span() is caller_span
    assert len(span_exporter.get_finished_spans()) == 1
    assert _no_context_errors(caplog)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["drain", "error", "caller_error"])
async def test_async_stream_drained_in_other_task(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
    ending: str,
) -> None:
    invocation = handler.inference("prov", _attach_to_context=False)
    error = ConnectionError("boom")

    async def chunks() -> AsyncIterator[str]:
        assert get_current_span() is invocation.span
        yield "a"
        if ending == "error":
            raise error
        yield "b"

    stream = _AsyncWrapper(chunks(), invocation)
    before = context.get_current()

    assert await asyncio.create_task(anext(stream)) == "a"
    assert context.get_current() is before

    if ending == "drain":
        assert [c async for c in stream] == ["b"]
    elif ending == "error":
        with pytest.raises(ConnectionError):
            await asyncio.create_task(anext(stream))
    else:
        with pytest.raises(ConnectionError):
            async with stream:
                assert get_current_span() is caller_span
                raise error

    assert context.get_current() is before
    (span,) = span_exporter.get_finished_spans()
    assert span.parent is not None
    assert span.parent.span_id == caller_span.get_span_context().span_id
    if ending != "drain":
        assert span.attributes is not None
        assert span.attributes["error.type"] == "ConnectionError"
    assert _no_context_errors(caplog)


def test_stream_built_later_under_another_span(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Deferred parse(): the wrapper is built after the call returned, under an unrelated span."""
    invocation = handler.inference("prov", _attach_to_context=False)

    with handler._tracer.start_as_current_span("parse") as parse_span:
        stream = _SyncWrapper(iter(["a", "b"]), invocation)
        assert next(stream) == "a"
        assert get_current_span() is parse_span

    assert get_current_span() is caller_span
    assert list(stream) == ["b"]
    assert stream._self_seen == [invocation.span, invocation.span]
    assert get_current_span() is caller_span
    assert invocation._finished
    assert _no_context_errors(caplog)


def test_activate_fails_on_escaping_error_and_never_stops(
    handler: TelemetryHandler,
    caller_span: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    invocation = handler.inference("prov", _attach_to_context=False)
    with invocation.activate():
        pass
    assert not invocation._finished

    with pytest.raises(StopIteration):
        with invocation.activate():
            raise StopIteration
    assert not invocation._finished

    with pytest.raises(ValueError):
        with invocation.activate():
            raise ValueError("boom")

    assert invocation._finished
    assert get_current_span() is caller_span
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes["error.type"] == "ValueError"


@pytest.mark.parametrize("finish", ["stop", "fail"])
def test_completion_hook_runs_with_span_current(
    span_exporter: InMemorySpanExporter, caller_span: Any, finish: str
) -> None:
    seen: list[Any] = []

    class Hook:
        def on_completion(self, **_kwargs: Any) -> None:
            seen.append(get_current_span())

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    handler = TelemetryHandler(
        tracer_provider=provider, completion_hook=Hook()
    )
    invocation = handler.inference("prov", _attach_to_context=False)

    if finish == "stop":
        invocation.stop()
    else:
        invocation.fail(ValueError("boom"))

    assert seen == [invocation.span]
    assert get_current_span() is caller_span
