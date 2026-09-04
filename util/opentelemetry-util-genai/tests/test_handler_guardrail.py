# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind
from opentelemetry.util.genai.handler import TelemetryHandler


def _make_span_exporter_and_handler() -> (
    tuple[InMemorySpanExporter, TelemetryHandler]
):
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return span_exporter, TelemetryHandler(tracer_provider=tracer_provider)


def test_guardrail_span_attributes_and_name() -> None:
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.guardrail(
        "content_filter",
        provider="openai",
    )
    invocation.stop()

    span = span_exporter.get_finished_spans()[0]
    assert span.name == "run_guardrail content_filter"
    assert span.kind == SpanKind.INTERNAL
    assert span.attributes == {
        "gen_ai.operation.name": "run_guardrail",
        "gen_ai.guardrail.component.name": "content_filter",
        "gen_ai.provider.name": "openai",
        "gen_ai.guardrail.verdict.type": "allow",
    }
    assert "gen_ai.guardrail.target.type" not in span.attributes


def test_guardrail_target_type_and_deny_verdict() -> None:
    span_exporter, handler = _make_span_exporter_and_handler()
    invocation = handler.guardrail(
        "jailbreak_filter",
        provider="openai",
        target_type="input",
    )
    invocation.triggered = True
    invocation.stop()

    span = span_exporter.get_finished_spans()[0]
    assert span.attributes["gen_ai.guardrail.verdict.type"] == "deny"
    assert span.attributes["gen_ai.guardrail.target.type"] == "input"
