# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Ollama streaming chat (inference)."""

from __future__ import annotations

from typing import Any

import ollama

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ..conftest import CHAT_MODEL, HOST
from .inference import NO_RESPONSE_ID


class InferenceStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = NO_RESPONSE_ID

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            OllamaInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("inference_streaming_conformance.yaml"):
                for _ in ollama.Client(host=HOST).chat(
                    model=CHAT_MODEL,
                    messages=[
                        {"role": "user", "content": "Say this is a test"}
                    ],
                    stream=True,
                ):
                    pass
