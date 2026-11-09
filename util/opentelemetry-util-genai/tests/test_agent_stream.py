# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Agent invocations streamed through the base stream wrappers."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import PurePosixPath

import pytest

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import get_current_span
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.utils import content_to_text


class _Telemetry:
    def __init__(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.metric_reader = InMemoryMetricReader()
        self.handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=MeterProvider(metric_readers=[self.metric_reader]),
        )
        self.tracer = tracer_provider.get_tracer(__name__)

    def span(self, name: str) -> ReadableSpan:
        spans = [
            s
            for s in self.span_exporter.get_finished_spans()
            if s.name == name
        ]
        assert len(spans) == 1, [
            s.name for s in self.span_exporter.get_finished_spans()
        ]
        return spans[0]

    def metric_names(self) -> set[str]:
        data = self.metric_reader.get_metrics_data()
        if data is None:
            return set()
        return {
            metric.name
            for resource_metrics in data.resource_metrics
            for scope_metrics in resource_metrics.scope_metrics
            for metric in scope_metrics.metrics
        }


@pytest.fixture(name="telemetry")
def fixture_telemetry() -> _Telemetry:
    return _Telemetry()


class _AgentStream(SyncStreamWrapper[object]):
    def __init__(self, stream, invocation):
        super().__init__(stream, invocation)
        invocation.suspend()

    def _process_chunk(self, chunk: object) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


class _AsyncAgentStream(AsyncStreamWrapper[object]):
    def __init__(self, stream, invocation):
        super().__init__(stream, invocation)
        invocation.suspend()

    def _process_chunk(self, chunk: object) -> None:
        pass

    def _on_stream_end(self) -> None:
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        self._self_invocation.fail(error)


def _chunks(*items: object) -> Iterator[object]:
    yield from items


async def _achunks(*items: object) -> AsyncIterator[object]:
    for item in items:
        yield item


def test_agent_stream_records_no_inference_stream_telemetry(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(_AgentStream(_chunks("a", "b"), invocation))

    span = telemetry.span("invoke_agent writer")
    assert GenAI.GEN_AI_REQUEST_STREAM not in (span.attributes or {})
    assert telemetry.metric_names() == {"gen_ai.invoke_agent.duration"}


def test_workflow_stream_records_no_inference_stream_telemetry(telemetry):
    invocation = telemetry.handler.workflow("pipeline")
    list(_AgentStream(_chunks("a"), invocation))

    span = telemetry.span("invoke_workflow pipeline")
    assert GenAI.GEN_AI_REQUEST_STREAM not in (span.attributes or {})


def test_sync_agent_stream_scopes_context_to_agent_code(telemetry):
    caller_span = get_current_span()
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    inside = []

    def agent_body() -> Iterator[str]:
        inside.append(get_current_span())
        yield "a"
        inside.append(get_current_span())
        yield "b"

    for _ in _AgentStream(agent_body(), invocation):
        assert get_current_span() is caller_span

    assert inside == [invocation.span, invocation.span]


def test_async_agent_stream_scopes_context_to_agent_code(telemetry):
    inside = []

    async def agent_body() -> AsyncIterator[str]:
        inside.append(get_current_span())
        yield "a"
        inside.append(get_current_span())
        yield "b"

    async def exercise() -> object:
        caller_span = get_current_span()
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        async for _ in _AsyncAgentStream(agent_body(), invocation):
            assert get_current_span() is caller_span
        return invocation.span

    agent_span = asyncio.run(exercise())
    assert inside == [agent_span, agent_span]


def test_sync_held_agent_stream_does_not_parent_caller_work(telemetry):
    def run() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        # Keep the stream referenced so it outlives the loop, as a caller
        # holding it would; otherwise it is finalized at ``break``.
        stream = _AgentStream(_chunks("a", "b"), invocation)
        for _ in stream:
            break
        with telemetry.tracer.start_as_current_span("caller-work"):
            pass
        del stream

    run()
    gc.collect()

    assert telemetry.span("caller-work").parent is None
    agent = telemetry.span("invoke_agent writer")
    assert (agent.attributes or {})[ERROR_TYPE] == "_OTHER"


def test_async_held_agent_stream_does_not_parent_caller_work(telemetry):
    async def exercise() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        # See test_sync_held_agent_stream_does_not_parent_caller_work.
        stream = _AsyncAgentStream(_achunks("a", "b"), invocation)
        async for _ in stream:
            break
        with telemetry.tracer.start_as_current_span("caller-work"):
            pass
        del stream

    asyncio.run(exercise())
    gc.collect()

    assert telemetry.span("caller-work").parent is None
    agent = telemetry.span("invoke_agent writer")
    assert (agent.attributes or {})[ERROR_TYPE] == "_OTHER"


def test_sync_stream_tells_exhaustion_from_early_close(telemetry):
    drained = _AgentStream(
        _chunks("a"), telemetry.handler.invoke_local_agent(agent_name="a")
    )
    list(drained)
    closed = _AgentStream(
        _chunks("a", "b"), telemetry.handler.invoke_local_agent(agent_name="b")
    )
    next(closed)
    closed.close()

    assert drained._self_exhausted is True
    assert closed._self_exhausted is False


def test_async_stream_tells_exhaustion_from_early_close(telemetry):
    async def exercise() -> tuple[bool, bool]:
        drained = _AsyncAgentStream(
            _achunks("a"), telemetry.handler.invoke_local_agent(agent_name="a")
        )
        async for _ in drained:
            pass
        closed = _AsyncAgentStream(
            _achunks("a", "b"),
            telemetry.handler.invoke_local_agent(agent_name="b"),
        )
        await closed.__anext__()
        await closed.aclose()
        return drained._self_exhausted, closed._self_exhausted

    assert asyncio.run(exercise()) == (True, False)


@dataclass
class _Answer:
    city: str
    temperature: int


class _Unserializable:
    def __str__(self) -> str:
        raise RuntimeError("no str")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("text", "text"),
        (None, None),
        (
            _Answer(city="Paris", temperature=21),
            '{"city":"Paris","temperature":21}',
        ),
        ({"k": "é"}, '{"k":"é"}'),
        (_Unserializable(), None),
        (Decimal("1.5"), "1.5"),
        (PurePosixPath("/tmp/a.txt"), "/tmp/a.txt"),
        ([Decimal("1.5")], '["1.5"]'),
    ],
)
def test_content_to_text(content, expected):
    assert content_to_text(content) == expected
