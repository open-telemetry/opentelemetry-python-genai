# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai generate_content."""

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
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class GenerateContentScenario(Scenario):
    expected_spans = ("generate_content",)
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
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("generate_content_conformance.yaml"):
                client = Client(
                    api_key="test_google_genai_api_key", vertexai=False
                )
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Say this is a test",
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
        assert response_ids == ["Ibo-aoj4EpSy1MkPsezpqQE"], (
            f"Expected gen_ai.response.id ['Ibo-aoj4EpSy1MkPsezpqQE'] but saw {response_ids}"
        )
