# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for CrewAI LLM inference events."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        source = SimpleNamespace(
            provider="openai",
            model="gpt-4.1-nano",
            base_url="https://api.openai.com/v1",
        )
        started = LLMCallStartedEvent(
            call_id="conformance-call",
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
            temperature=0.0,
            max_tokens=32,
        )
        completed = LLMCallCompletedEvent(
            call_id="conformance-call",
            model="gpt-4.1-nano",
            started_event_id=started.event_id,
            response=SimpleNamespace(
                content="2 + 2 equals 4.",
                model="gpt-4.1-nano-2025-04-14",
                id="chatcmpl-conformance",
                finish_reason="stop",
            ),
            call_type=LLMCallType.LLM_CALL,
            usage={"prompt_tokens": 12, "completion_tokens": 8},
            finish_reason="stop",
            response_id="chatcmpl-conformance",
        )

        with instrument(
            CrewAIInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="SPAN_ONLY",
        ):
            start_future = crewai_event_bus.emit(source, started)
            if start_future is not None:
                start_future.result(timeout=5)
            completed_future = crewai_event_bus.emit(source, completed)
            if completed_future is not None:
                completed_future.result(timeout=5)
