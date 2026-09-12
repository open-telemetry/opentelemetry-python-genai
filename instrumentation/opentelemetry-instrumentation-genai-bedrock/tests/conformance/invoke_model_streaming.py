# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_model_with_response_stream (streaming chat)."""

from __future__ import annotations

import json
from typing import Any

import boto3

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class InvokeModelStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
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
        with instrument(
            BedrockInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "test_invoke_model_stream_titan_conformance.yaml"
            ):
                client = boto3.client(
                    "bedrock-runtime", region_name="us-east-1"
                )
                response = client.invoke_model_with_response_stream(
                    modelId="amazon.titan-text-express-v1",
                    body=json.dumps(
                        {
                            "inputText": "Say this is a test",
                            "textGenerationConfig": {
                                "maxTokenCount": 10,
                                "temperature": 0.8,
                                "topP": 1.0,
                                "stopSequences": ["|"],
                            },
                        }
                    ),
                )
                body = response.get("body")
                if body:
                    for _ in body:
                        pass
