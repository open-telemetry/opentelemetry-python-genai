# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai-v2 chat completion (inference)."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = ("chat",)
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
        with instrument(
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            semconv="gen_ai_latest_experimental",
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("inference_conformance.yaml"):
                OpenAI().chat.completions.create(
                    messages=[
                        {"role": "user", "content": "Say this is a test"}
                    ],
                    model="gpt-4o-mini",
                    stream=False,
                )
