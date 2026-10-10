# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentelemetry.instrumentation.genai.qwenpaw.patch import _build_invocation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import LocalAgentInvocation


@pytest.mark.parametrize("cached_name", [None, "cached-agent"])
def test_agent_name_does_not_load_configuration(
    cached_name: str | None,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    class Runner:
        _agent_name: str | None = cached_name

        @property
        def agent_name(self) -> str:
            raise AssertionError("agent_name must not load configuration")

    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = _build_invocation(handler, Runner(), None, None)
    assert isinstance(invocation, LocalAgentInvocation)
    invocation.stop()

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes.get(GenAI.GEN_AI_AGENT_NAME) == cached_name
    assert span.name == (
        f"invoke_agent {cached_name}" if cached_name else "invoke_agent"
    )


def test_disabled_content_capture_does_not_read_messages(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    class Message:
        role: str = "user"

        def get_text_content(self) -> str:
            raise AssertionError("message content must not be read")

    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "NO_CONTENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = _build_invocation(
        handler, SimpleNamespace(), [Message()], None
    )
    invocation.stop()

    (span,) = span_exporter.get_finished_spans()
    assert GenAI.GEN_AI_INPUT_MESSAGES not in span.attributes
