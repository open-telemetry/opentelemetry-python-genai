# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic beta chat (inference)."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from anthropic import Anthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceBetaScenario(Scenario):
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
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("inference_beta_conformance.yaml"):
                    Anthropic().beta.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=100,
                        messages=[
                            {
                                "role": "user",
                                "content": "Say hello in one word.",
                            }
                        ],
                    )


class InferenceBetaStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        stream_values = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.request.stream"
        ]
        assert stream_values == [True], (
            "streaming messages should set gen_ai.request.stream=true on the "
            f"chat span; saw {stream_values}"
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
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette(
                    "inference_beta_streaming_conformance.yaml"
                ):
                    with Anthropic().beta.messages.create(
                        model="claude-sonnet-4-6",
                        max_tokens=100,
                        messages=[
                            {
                                "role": "user",
                                "content": "Say hello in one word.",
                            }
                        ],
                        stream=True,
                    ) as stream:
                        for _ in stream:
                            pass
