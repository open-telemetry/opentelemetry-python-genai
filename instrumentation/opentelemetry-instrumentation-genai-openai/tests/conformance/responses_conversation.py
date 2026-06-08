# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API multi-turn conversation."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

DEFAULT_MODEL = "gpt-4o-mini"


class ResponsesConversationScenario(Scenario):
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
            with vcr.use_cassette("responses_conversation_conformance.yaml"):
                client = OpenAI()
                first = client.responses.create(
                    model=DEFAULT_MODEL,
                    input="Remember that my favorite color is blue.",
                )

                client.responses.create(
                    model=DEFAULT_MODEL,
                    input="What is my favorite color?",
                    previous_response_id=first.id,
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        operations = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.operation.name"
        ]
        assert operations == ["chat", "chat"], (
            "Responses conversation exercises two Responses API calls "
            f"(initial request and follow-up); saw spans {operations}"
        )
