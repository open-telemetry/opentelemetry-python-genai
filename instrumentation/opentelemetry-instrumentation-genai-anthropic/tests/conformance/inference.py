# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic chat (inference)."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from anthropic import Anthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
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
        key_override = (
            {}
            if os.getenv("ANTHROPIC_API_KEY")
            else {"ANTHROPIC_API_KEY": "test_anthropic_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                semconv="gen_ai_latest_experimental",
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("inference_conformance.yaml"):
                    Anthropic().messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=100,
                        messages=[
                            {
                                "role": "user",
                                "content": "Say hello in one word.",
                            }
                        ],
                    )
