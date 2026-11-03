# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import gc
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from unittest.mock import patch

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
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.stream import (
    AsyncAgentStreamWrapper,
    SyncAgentStreamWrapper,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    OutputMessage,
    TextPart,
)

_INFERENCE_STREAM_METRICS = {
    "gen_ai.client.operation.time_to_first_chunk",
    "gen_ai.client.operation.time_per_output_chunk",
}


class _Telemetry:
    def __init__(self, mode: ContentCapturingMode) -> None:
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.metric_reader = InMemoryMetricReader()
        with patch(
            "opentelemetry.util.genai.handler.get_content_capturing_mode",
            return_value=mode,
        ):
            self.handler = TelemetryHandler(
                tracer_provider=tracer_provider,
                meter_provider=MeterProvider(
                    metric_readers=[self.metric_reader]
                ),
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
    return _Telemetry(ContentCapturingMode.SPAN_ONLY)


@pytest.fixture(name="telemetry_no_content")
def fixture_telemetry_no_content() -> _Telemetry:
    return _Telemetry(ContentCapturingMode.NO_CONTENT)


def _output_messages(span: ReadableSpan) -> list[dict[str, object]]:
    value = (span.attributes or {})[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert isinstance(value, str)
    return [
        {key: val for key, val in message.items() if val is not None}
        for message in json.loads(value)
    ]


def _assistant_text(text: str) -> list[dict[str, object]]:
    return [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": text}],
        }
    ]


class _DeltaStream(SyncAgentStreamWrapper[object]):
    def __init__(self, stream, invocation):
        super().__init__(stream, invocation)
        self._self_seen: list[object] = []

    def _process_chunk(self, chunk: object) -> None:
        self._self_seen.append(chunk)
        self._add_output(chunk)


class _AsyncDeltaStream(AsyncAgentStreamWrapper[object]):
    def _process_chunk(self, chunk: object) -> None:
        self._add_output(chunk)


class _FinalStream(SyncAgentStreamWrapper[tuple[str, bool]]):
    def _process_chunk(self, chunk: tuple[str, bool]) -> None:
        text, is_final = chunk
        if is_final:
            self._set_output(text)
        else:
            self._add_output(text)


class _MessagesStream(SyncAgentStreamWrapper[str]):
    def _process_chunk(self, chunk: str) -> None:
        self._add_output(chunk)
        self._set_output_messages(
            [
                OutputMessage(
                    role="assistant", parts=[TextPart(content=chunk.upper())]
                )
            ]
        )


class _UsageStream(SyncAgentStreamWrapper[str]):
    def _process_chunk(self, chunk: str) -> None:
        self._add_output(chunk)

    def _on_stream_end(self) -> None:
        invocation = self._self_agent_invocation
        assert isinstance(invocation, AgentInvocation)
        invocation.finish_reasons = ["stop"]
        super()._on_stream_end()


@dataclass
class _Answer:
    city: str
    temperature: int


def _chunks(*items: object) -> Iterator[object]:
    yield from items


async def _achunks(*items: object) -> AsyncIterator[object]:
    for item in items:
        yield item


def test_sync_agent_stream_joins_streamed_output(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    wrapper = _DeltaStream(_chunks("Hel", "lo", "!"), invocation)

    assert list(wrapper) == ["Hel", "lo", "!"]

    span = telemetry.span("invoke_agent writer")
    assert span.status.status_code == StatusCode.UNSET
    assert _output_messages(span) == _assistant_text("Hello!")
    assert GenAI.GEN_AI_REQUEST_STREAM not in (span.attributes or {})


def test_sync_agent_stream_records_no_inference_stream_metrics(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(_DeltaStream(_chunks("a", "b"), invocation))

    assert telemetry.metric_names() == {"gen_ai.invoke_agent.duration"}


def test_sync_agent_stream_serializes_structured_output(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(
        _DeltaStream(
            _chunks(_Answer(city="Paris", temperature=21)), invocation
        )
    )

    span = telemetry.span("invoke_agent writer")
    (message,) = _output_messages(span)
    (part,) = message["parts"]
    assert part["type"] == "text"
    assert json.loads(part["content"]) == {"city": "Paris", "temperature": 21}


def test_sync_agent_stream_final_output_replaces_streamed_pieces(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(
        _FinalStream(
            _chunks(("Hel", False), ("lo", False), ("Hello there", True)),
            invocation,
        )
    )

    span = telemetry.span("invoke_agent writer")
    assert _output_messages(span) == _assistant_text("Hello there")


def test_sync_agent_stream_output_messages_take_precedence(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(_MessagesStream(_chunks("a", "b"), invocation))

    span = telemetry.span("invoke_agent writer")
    assert _output_messages(span) == _assistant_text("B")


def test_sync_agent_stream_subclass_can_extend_finalization(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(_UsageStream(_chunks("ok"), invocation))

    attrs = telemetry.span("invoke_agent writer").attributes or {}
    assert attrs[GenAI.GEN_AI_RESPONSE_FINISH_REASONS] == ("stop",)
    assert _output_messages(
        telemetry.span("invoke_agent writer")
    ) == _assistant_text("ok")


def test_sync_agent_stream_skips_output_when_content_disabled(
    telemetry_no_content,
):
    telemetry = telemetry_no_content
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    wrapper = _DeltaStream(_chunks("a", "b"), invocation)

    assert list(wrapper) == ["a", "b"]
    assert wrapper._self_seen == ["a", "b"]
    assert wrapper._self_output_parts == []

    attrs = telemetry.span("invoke_agent writer").attributes or {}
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


def test_sync_agent_stream_empty_output_sets_no_messages(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    list(_DeltaStream(_chunks(), invocation))

    attrs = telemetry.span("invoke_agent writer").attributes or {}
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attrs


def test_sync_agent_stream_error_records_partial_output(telemetry):
    def failing() -> Iterator[str]:
        yield "partial"
        raise ConnectionError("dropped")

    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    wrapper = _DeltaStream(failing(), invocation)

    with pytest.raises(ConnectionError, match="dropped"):
        list(wrapper)

    span = telemetry.span("invoke_agent writer")
    assert span.status.status_code == StatusCode.ERROR
    assert (span.attributes or {})[ERROR_TYPE] == "ConnectionError"
    assert _output_messages(span) == _assistant_text("partial")


def test_sync_agent_stream_caller_error_in_with_block(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")

    with pytest.raises(ValueError, match="caller"):
        with _DeltaStream(_chunks("a", "b"), invocation) as wrapper:
            next(wrapper)
            raise ValueError("caller")

    span = telemetry.span("invoke_agent writer")
    assert (span.attributes or {})[ERROR_TYPE] == "ValueError"
    assert _output_messages(span) == _assistant_text("a")


def test_sync_agent_stream_close_finalizes_successfully(telemetry):
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    wrapper = _DeltaStream(_chunks("a", "b"), invocation)
    next(wrapper)
    wrapper.close()

    span = telemetry.span("invoke_agent writer")
    assert span.status.status_code == StatusCode.UNSET
    assert _output_messages(span) == _assistant_text("a")


def test_sync_agent_stream_scopes_context_to_agent_code(telemetry):
    caller_span = get_current_span()
    invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
    inside = []

    def agent_body() -> Iterator[str]:
        inside.append(get_current_span())
        yield "a"
        inside.append(get_current_span())
        yield "b"

    wrapper = _DeltaStream(agent_body(), invocation)
    assert get_current_span() is caller_span

    for _ in wrapper:
        assert get_current_span() is caller_span

    assert inside == [invocation.span, invocation.span]


def test_sync_agent_stream_abandoned_does_not_parent_caller_work(telemetry):
    def run() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        for _ in _DeltaStream(_chunks("a", "b"), invocation):
            break
        with telemetry.tracer.start_as_current_span("caller-work"):
            pass

    run()
    gc.collect()

    agent = telemetry.span("invoke_agent writer")
    assert telemetry.span("caller-work").parent is None
    assert agent.status.status_code == StatusCode.ERROR
    assert (agent.attributes or {})[ERROR_TYPE] == "_OTHER"
    assert _output_messages(agent) == _assistant_text("a")


def test_sync_agent_stream_supports_workflow_invocation(telemetry):
    invocation = telemetry.handler.workflow("pipeline")
    assert isinstance(invocation, WorkflowInvocation)
    list(_DeltaStream(_chunks("done"), invocation))

    span = telemetry.span("invoke_workflow pipeline")
    assert _output_messages(span) == _assistant_text("done")


def test_async_agent_stream_joins_streamed_output(telemetry):
    async def exercise() -> list[object]:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        wrapper = _AsyncDeltaStream(_achunks("Hel", "lo"), invocation)
        return [chunk async for chunk in wrapper]

    assert asyncio.run(exercise()) == ["Hel", "lo"]

    span = telemetry.span("invoke_agent writer")
    assert span.status.status_code == StatusCode.UNSET
    assert _output_messages(span) == _assistant_text("Hello")
    assert not telemetry.metric_names() & _INFERENCE_STREAM_METRICS


def test_async_agent_stream_aclose_finalizes_successfully(telemetry):
    async def exercise() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        wrapper = _AsyncDeltaStream(_achunks("a", "b"), invocation)
        await wrapper.__anext__()
        await wrapper.aclose()

    asyncio.run(exercise())

    span = telemetry.span("invoke_agent writer")
    assert span.status.status_code == StatusCode.UNSET
    assert _output_messages(span) == _assistant_text("a")


def test_async_agent_stream_error_records_partial_output(telemetry):
    async def failing() -> AsyncIterator[str]:
        yield "partial"
        raise ConnectionError("dropped")

    async def exercise() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        async for _ in _AsyncDeltaStream(failing(), invocation):
            pass

    with pytest.raises(ConnectionError, match="dropped"):
        asyncio.run(exercise())

    span = telemetry.span("invoke_agent writer")
    assert (span.attributes or {})[ERROR_TYPE] == "ConnectionError"
    assert _output_messages(span) == _assistant_text("partial")


def test_async_agent_stream_caller_error_in_async_with_block(telemetry):
    async def exercise() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        async with _AsyncDeltaStream(_achunks("a", "b"), invocation) as w:
            await w.__anext__()
            raise ValueError("caller")

    with pytest.raises(ValueError, match="caller"):
        asyncio.run(exercise())

    span = telemetry.span("invoke_agent writer")
    assert (span.attributes or {})[ERROR_TYPE] == "ValueError"
    assert _output_messages(span) == _assistant_text("a")


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
        wrapper = _AsyncDeltaStream(agent_body(), invocation)
        assert get_current_span() is caller_span
        async for _ in wrapper:
            assert get_current_span() is caller_span
        return invocation.span

    agent_span = asyncio.run(exercise())
    assert inside == [agent_span, agent_span]


def test_async_agent_stream_abandoned_does_not_parent_caller_work(telemetry):
    async def exercise() -> None:
        invocation = telemetry.handler.invoke_local_agent(agent_name="writer")
        async for _ in _AsyncDeltaStream(_achunks("a", "b"), invocation):
            break
        with telemetry.tracer.start_as_current_span("caller-work"):
            pass

    asyncio.run(exercise())
    gc.collect()

    agent = telemetry.span("invoke_agent writer")
    assert telemetry.span("caller-work").parent is None
    assert (agent.attributes or {})[ERROR_TYPE] == "_OTHER"
