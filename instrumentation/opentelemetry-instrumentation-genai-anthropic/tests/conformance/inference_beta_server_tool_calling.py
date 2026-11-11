# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Anthropic beta chat with server-side tool calls."""

from __future__ import annotations

import json
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


class InferenceBetaServerToolCallingScenario(Scenario):
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
        parts = output_messages[0][0]["parts"]
        calls = [p for p in parts if p["type"] == "server_tool_call"]
        responses = [
            p for p in parts if p["type"] == "server_tool_call_response"
        ]
        assert len(calls) == 1
        assert calls[0] == {
            "type": "server_tool_call",
            "id": "srvtoolu_01PKGW51KPFgUP7K4kSeMtPw",
            "name": "web_search",
            "server_tool_call": {
                "type": "web_search",
                "input": {"query": "OpenTelemetry"},
            },
        }
        assert len(responses) == 1
        assert responses[0]["id"] == "srvtoolu_01PKGW51KPFgUP7K4kSeMtPw"
        assert (
            responses[0]["server_tool_call_response"]["type"] == "web_search"
        )
        assert responses[0]["server_tool_call_response"]["content"]

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
                    "inference_beta_server_tool_calling_conformance.yaml"
                ):
                    Anthropic().beta.messages.create(
                        model="claude-sonnet-5-5",
                        max_tokens=256,
                        messages=[
                            {
                                "role": "user",
                                "content": "Search for OpenTelemetry.",
                            }
                        ],
                        tools=[
                            {
                                "type": "web_search_20250305",
                                "name": "web_search",
                                "max_uses": 1,
                            }
                        ],
                    )
