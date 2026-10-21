# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_agent."""

from __future__ import annotations

from typing import Any

import boto3

from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InvokeAgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

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
            with vcr.use_cassette("test_invoke_agent_conformance.yaml"):
                client = boto3.client(
                    "bedrock-agent-runtime", region_name="us-east-1"
                )
                response = client.invoke_agent(
                    agentId="AGENT1234567",
                    agentAliasId="ALIAS1234567",
                    sessionId="conformance-session-1",
                    inputText="What is the capital of France?",
                )
                for _ in response["completion"]:
                    pass
