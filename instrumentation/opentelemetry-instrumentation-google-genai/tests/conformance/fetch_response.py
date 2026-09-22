# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Google GenAI interaction retrieval."""

from __future__ import annotations

import json
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


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]


class FetchResponseScenario(Scenario):
    expected_spans = {"fetch_response": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name")
            == "fetch_response"
        ]
        assert len(spans) == 1, "expected exactly one fetch_response span"
        span = spans[0]

        assert _attr(span, "gen_ai.response.id") == "interaction-fetch-1"
        assert _attr(span, "gen_ai.response.status") == "completed"
        assert _part_types(_attr(span, "gen_ai.output.messages")) == ["text"]
        # A fetch performs no inference: the fetched interaction's token counts
        # belong to the original generation and must not be reported here.
        for name in (
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ):
            assert _attr(span, name) is None, f"{name} must not be recorded"
        # A fetched response carries no record of the original request's input.
        assert _attr(span, "gen_ai.input.messages") is None

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        from opentelemetry.test_util_genai.instrumentor import instrument

        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "interactions_fetch_response_conformance.yaml"
            ):
                client = Client(
                    api_key="test_google_genai_api_key", vertexai=False
                )
                client.interactions.get("interaction-fetch-1")
