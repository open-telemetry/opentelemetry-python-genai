# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: Google GenAI server-side tool calls."""

from __future__ import annotations

import json
import os
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


def _output_parts(report: LiveCheckReport) -> list[dict[str, Any]]:
    output_messages = [
        json.loads(attribute["value"])
        for entry in report["samples"]
        if "span" in entry
        for attribute in entry["span"]["attributes"]
        if attribute["name"] == "gen_ai.output.messages"
    ]
    assert len(output_messages) == 1
    return output_messages[0][0]["parts"]


class GenerateContentServerToolCallingScenario(Scenario):
    expected_spans = {"generate_content": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        assert _output_parts(report)[:2] == [
            {
                "name": "code_execution",
                "server_tool_call": {
                    "code": 'print("one")\n',
                    "language": "PYTHON",
                    "type": "code_execution",
                },
                "id": None,
                "type": "server_tool_call",
            },
            {
                "server_tool_call_response": {
                    "outcome": "OUTCOME_OK",
                    "output": "one\n",
                    "type": "code_execution",
                },
                "id": None,
                "type": "server_tool_call_response",
            },
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
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "generate_content_server_tool_calling_conformance.yaml"
            ):
                client = Client(
                    api_key=os.getenv(
                        "GOOGLE_API_KEY", "test_google_genai_api_key"
                    ),
                    vertexai=False,
                )
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Run Python to print one.",
                    config={"tools": [{"code_execution": {}}]},
                )


class InteractionsServerToolCallingScenario(Scenario):
    expected_spans = {"interactions.create": 1}
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

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        parts = _output_parts(report)
        calls = [p for p in parts if p["type"] == "server_tool_call"]
        responses = [
            p for p in parts if p["type"] == "server_tool_call_response"
        ]
        assert len(calls) == 1
        assert calls[0]["name"] == "google_search"
        assert calls[0]["server_tool_call"]["type"] == "google_search"
        assert calls[0]["server_tool_call"]["search_type"] == "web_search"
        assert calls[0]["server_tool_call"]["arguments"]["queries"]
        assert len(responses) == 1
        assert responses[0]["id"] == calls[0]["id"]
        assert (
            responses[0]["server_tool_call_response"]["type"]
            == "google_search"
        )
        assert responses[0]["server_tool_call_response"]["is_error"] is False
        assert isinstance(
            responses[0]["server_tool_call_response"]["result"], list
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
            with vcr.use_cassette(
                "interactions_server_tool_calling_conformance.yaml"
            ):
                client = Client(
                    api_key=os.getenv(
                        "GOOGLE_API_KEY", "test_google_genai_api_key"
                    ),
                    vertexai=False,
                )
                client.interactions.create(
                    model="gemini-2.5-flash",
                    input="Search for OpenTelemetry.",
                    tools=[{"type": "google_search"}],
                )
