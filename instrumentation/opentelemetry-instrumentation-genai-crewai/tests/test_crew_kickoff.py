# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""VCR-backed integration test: a real Agent.kickoff() drives the listener.

Mirrors the ``kickoff_agent`` scenario from the donated OpenInference CrewAI
tests and replays its genuinely-recorded ``inference_conformance.yaml``
cassette (shared with the conformance scenario). Unlike
``test_event_listener.py``, events here arise from real CrewAI execution —
including its thread-pool event dispatch — so spans are awaited briefly
before asserting.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from crewai import LLM, Agent

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.trace import SpanKind, StatusCode

_SPAN_WAIT_SECONDS = 5.0


def _wait_for_spans(
    span_exporter: InMemorySpanExporter, count: int
) -> tuple[ReadableSpan, ...]:
    """Wait for CrewAI's thread-pool event dispatch to finish the spans."""
    deadline = time.monotonic() + _SPAN_WAIT_SECONDS
    while time.monotonic() < deadline:
        spans = span_exporter.get_finished_spans()
        if len(spans) >= count:
            return spans
        time.sleep(0.05)
    return span_exporter.get_finished_spans()


@pytest.fixture(autouse=True)
def crewai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_api_key")


def test_agent_kickoff_emits_chat_span(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    vcr: Any,
) -> None:
    agent = Agent(
        role="Helpful Assistant",
        goal="Answer questions clearly and concisely",
        backstory="You are a helpful assistant.",
        allow_delegation=False,
        llm=LLM(model="gpt-4.1-nano", temperature=0),
    )

    with vcr.use_cassette("inference_conformance.yaml"):
        result = agent.kickoff("What is 2+2?")

    assert "4" in str(getattr(result, "raw", result))

    spans = _wait_for_spans(span_exporter, count=1)
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gpt-4.1-nano"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    attributes = span.attributes
    assert attributes is not None
    assert (
        attributes[GenAI.GEN_AI_OPERATION_NAME]
        == GenAI.GenAiOperationNameValues.CHAT.value
    )
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "crewai"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4.1-nano"
    # Values recorded in the OpenInference cassette.
    assert (
        attributes[GenAI.GEN_AI_RESPONSE_ID]
        == "chatcmpl-DFqnawCI6sxrV5JpTQf8hYUeKY9fG"
    )
    assert isinstance(attributes[GenAI.GEN_AI_RESPONSE_ID], str)
    assert tuple(attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS]) == ("stop",)
    assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 50
    assert isinstance(attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS], int)
    assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert isinstance(attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS], int)

    input_messages = attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert isinstance(input_messages, str)
    assert "What is 2+2?" in input_messages
    output_messages = attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert isinstance(output_messages, str)
    assert "2 + 2 equals 4." in output_messages

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metric_names = {
        metric.name
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION in metric_names
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE in metric_names
