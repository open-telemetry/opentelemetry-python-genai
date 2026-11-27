# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: ``interactions.create`` across automatic continuation.

Each cassette holds two hops: the first stops with ``requires_continuation``
and the SDK resumes it from the continuation token in a second request.
"""

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

# Usage summed across both hops in the cassettes.
_EXPECTED_ATTRIBUTES: dict[str, object] = {
    "gen_ai.usage.input_tokens": 32,
    "gen_ai.usage.output_tokens": 19,
    "gen_ai.response.id": "v1_continuation_hop_2",
}


def _span_attributes(report: LiveCheckReport) -> dict[str, object]:
    return {
        attr["name"]: attr["value"]
        for entry in report["samples"]
        if "span" in entry
        for attr in entry["span"]["attributes"]
    }


class _ContinuationScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    cassette: str
    stream: bool

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
            with vcr.use_cassette(self.cassette):
                client = Client(
                    api_key="test_google_genai_api_key", vertexai=False
                )
                result = client.interactions.create(
                    model="gemini-2.5-flash",
                    input="What is OpenTelemetry?",
                    stream=self.stream,
                )
                if self.stream:
                    for _ in result:
                        pass

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        attributes = _span_attributes(report)
        for name, expected in _EXPECTED_ATTRIBUTES.items():
            assert attributes.get(name) == expected, (
                f"{name}: expected {expected!r}, saw {attributes.get(name)!r}"
            )


class ContinuationScenario(_ContinuationScenario):
    cassette = "interactions_continuation_conformance.yaml"
    stream = False


class ContinuationStreamingScenario(_ContinuationScenario):
    cassette = "interactions_continuation_streaming_conformance.yaml"
    stream = True

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        assert _span_attributes(report).get("gen_ai.request.stream") is True
