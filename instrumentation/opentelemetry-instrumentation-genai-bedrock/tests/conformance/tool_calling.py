# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock chat with tool calls."""

from __future__ import annotations

from typing import Any

import boto3

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class ToolCallingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.id",
        ),
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.model",
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
            BedrockInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("test_converse_tool_call_with_content.yaml"):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": (
                                    "What is the weather in Seattle and San"
                                    " Francisco today?"
                                )
                            }
                        ],
                    }
                ]
                tool_config = {
                    "tools": [
                        {
                            "toolSpec": {
                                "name": "get_current_weather",
                                "description": (
                                    "Get the current weather in a given location."
                                ),
                                "inputSchema": {
                                    "json": {
                                        "type": "object",
                                        "properties": {
                                            "location": {
                                                "type": "string",
                                                "description": (
                                                    "The name of the city"
                                                ),
                                            }
                                        },
                                        "required": ["location"],
                                    }
                                },
                            }
                        }
                    ]
                }
                client = boto3.client(
                    "bedrock-runtime", region_name="us-east-1"
                )
                client.converse(
                    messages=messages,
                    modelId="amazon.nova-micro-v1:0",
                    toolConfig=tool_config,
                )
