# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai chat completion (inference)."""

from __future__ import annotations

from typing import Any

from google.genai import Client

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = ("interactions.create",)
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_operation_name_unknown",
            message_substring="interactions.create",
        ),
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
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("inference_conformance.yaml"):
                client = Client(
                    api_key="test_google_genai_api_key", vertexai=False
                )
                client.interactions.create(
                    model="gemini-2.5-flash",
                    input="Hello, how can you help me today?",
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        response_ids = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.response.id"
        ]
        assert response_ids == [
            "v1_ChdMaWM4YXF2ekNMQ1k5TW9QLUpHZ3dBYxIXTGljOGFxdnpDTENZOU1vUC1KR2d3QWM"
        ], (
            "Expected gen_ai.response.id "
            "['v1_ChdMaWM4YXF2ekNMQ1k5TW9QLUpHZ3dBYxIXTGljOGFxdnpDTENZOU1vUC1KR2d3QWM'] "
            f"but saw {response_ids}"
        )
