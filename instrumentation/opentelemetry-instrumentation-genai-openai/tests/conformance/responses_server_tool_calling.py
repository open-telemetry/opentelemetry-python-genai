# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API hosted tool calling."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class ResponsesServerToolCallingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        output_messages = [
            json.loads(attribute["value"])
            for entry in report["samples"]
            if "span" in entry
            for attribute in entry["span"]["attributes"]
            if attribute["name"] == "gen_ai.output.messages"
        ]
        assert len(output_messages) == 1
        assert output_messages[0][0]["parts"] == [
            {
                "name": "web_search",
                "server_tool_call": {
                    "action": {
                        "query": "OpenTelemetry",
                        "type": "search",
                    },
                    "status": "completed",
                    "type": "web_search",
                },
                "id": "ws_1",
                "type": "server_tool_call",
            }
        ]

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
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "responses_server_tool_calling_conformance.yaml"
            ):
                OpenAI().responses.create(
                    model="gpt-4o-mini",
                    input="Search for OpenTelemetry.",
                    tools=[{"type": "web_search"}],
                )
