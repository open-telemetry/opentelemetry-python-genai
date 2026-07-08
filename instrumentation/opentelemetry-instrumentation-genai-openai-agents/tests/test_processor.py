# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
from typing import Any
from unittest.mock import MagicMock

import pytest
from agents.tracing.span_data import (
    AgentSpanData,
    FunctionSpanData,
    GenerationSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

from opentelemetry.instrumentation.genai.openai_agents.processor import (
    GenAITracingProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import ToolInvocation


class _Span:
    """Minimal stand-in for agents-library Span (must be weakref-able)."""

    def __init__(self, span_data: Any) -> None:
        self.span_data = span_data
        # Span objects in tests don't need a span_id since the processor
        # keys its WeakKeyDictionary by the Span instance itself, but
        # keep one around for parity with the real class.
        self.span_id = f"span-{id(self)}"


class _Trace:
    """Minimal stand-in for agents-library Trace."""

    def __init__(self, trace_id: str, name: str) -> None:
        self.trace_id = trace_id
        self.name = name


def _build_handler() -> MagicMock:
    return MagicMock()


def test_trace_start_end_creates_and_stops_workflow() -> None:
    handler = _build_handler()
    handler.workflow.return_value = MagicMock(attributes={})
    processor = GenAITracingProcessor(handler, provider="openai")
    trace = _Trace("trace-1", "Agent workflow")

    processor.on_trace_start(trace)
    handler.workflow.assert_called_once_with(name="Agent workflow")
    workflow_invocation = handler.workflow.return_value
    assert (
        workflow_invocation.attributes["gen_ai.workflow.name"]
        == "Agent workflow"
    )

    processor.on_trace_end(trace)
    workflow_invocation.stop.assert_called_once_with()


def test_agent_span_creates_invoke_local_agent() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(AgentSpanData(name="triage"))

    processor.on_span_start(span)
    handler.invoke_local_agent.assert_called_once_with(agent_name="triage")

    processor.on_span_end(span)
    handler.invoke_local_agent.return_value.stop.assert_called_once_with()


def test_function_span_creates_tool_invocation_and_sets_provider_metric() -> (
    None
):
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation, metric_attributes={}
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(
        FunctionSpanData(
            name="get_weather",
            input='{"city":"BCN"}',
            output=None,
        )
    )

    processor.on_span_start(span)
    handler.tool.assert_called_once_with(
        name="get_weather",
        tool_type="function",
    )
    tool_invocation = handler.tool.return_value

    assert tool_invocation.arguments == '{"city":"BCN"}'
    assert (
        tool_invocation.metric_attributes["gen_ai.provider.name"] == "openai"
    )

    # Output gets populated on the agents library span_data after the
    # tool runs; our on_span_end reads it.
    span.span_data.output = "sunny"
    processor.on_span_end(span)
    assert tool_invocation.tool_result == "sunny"
    tool_invocation.stop.assert_called_once_with()


def test_function_span_without_output_still_stops() -> None:
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation, metric_attributes={}
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(FunctionSpanData(name="noop", input=None, output=None))

    processor.on_span_start(span)
    processor.on_span_end(span)
    tool_invocation = handler.tool.return_value
    # tool_result stays as whatever MagicMock default; what matters is
    # we didn't crash and we stopped.
    tool_invocation.stop.assert_called_once_with()


def test_generation_and_response_spans_ignored() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")

    for span_data in (
        GenerationSpanData(model="gpt-4o-mini"),
        ResponseSpanData(),
    ):
        span = _Span(span_data)
        processor.on_span_start(span)
        processor.on_span_end(span)

    handler.invoke_local_agent.assert_not_called()
    handler.tool.assert_not_called()
    handler.inference.assert_not_called()


def test_handoff_emits_raw_span() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(
        HandoffSpanData(from_agent="triage", to_agent="weather_specialist")
    )
    # Doesn't raise; the actual OTel span emission is verified end-to-end
    # by the conformance scenario.
    processor.on_span_start(span)
    processor.on_span_end(span)


def test_shutdown_stops_open_invocations() -> None:
    handler = _build_handler()
    handler.tool.return_value = MagicMock(
        spec=ToolInvocation, metric_attributes={}
    )
    processor = GenAITracingProcessor(handler, provider="openai")
    # Hold strong references to the trace / span objects so the
    # WeakKeyDictionary entries survive until shutdown runs.
    trace = _Trace("t", "wf")
    agent_span = _Span(AgentSpanData(name="agent"))
    tool_span = _Span(
        FunctionSpanData(name="get_weather", input=None, output=None)
    )
    processor.on_trace_start(trace)
    processor.on_span_start(agent_span)
    processor.on_span_start(tool_span)
    assert len(processor._invocations) == 3

    processor.shutdown()

    handler.workflow.return_value.stop.assert_called_once_with()
    handler.invoke_local_agent.return_value.stop.assert_called_once_with()
    handler.tool.return_value.stop.assert_called_once_with()
    assert len(processor._invocations) == 0


def test_no_content_captured_when_capture_env_unset(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, raising=False
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    processor = GenAITracingProcessor(handler, provider="openai")
    span = _Span(
        FunctionSpanData(
            name="get_weather",
            input='{"city":"Barcelona"}',
            output="sunny",
        )
    )

    processor.on_span_start(span)
    processor.on_span_end(span)

    (tool_span,) = span_exporter.get_finished_spans()
    assert tool_span.attributes is not None
    assert "gen_ai.tool.call.arguments" not in tool_span.attributes
    assert "gen_ai.tool.call.result" not in tool_span.attributes
    # The non-content tool attributes are still present.
    assert tool_span.attributes["gen_ai.tool.name"] == "get_weather"


def test_state_uses_weakref_so_dropped_spans_are_collected() -> None:
    handler = _build_handler()
    processor = GenAITracingProcessor(handler, provider="openai")
    span: _Span | None = _Span(AgentSpanData(name="triage"))
    processor.on_span_start(span)
    assert len(processor._invocations) == 1

    span = None  # drop the only strong reference
    gc.collect()

    assert len(processor._invocations) == 0
