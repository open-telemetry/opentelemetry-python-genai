# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: CrewAI LLM callback events emit chat spans.

Mirrors the ``kickoff_agent`` scenario from the donated OpenInference CrewAI
tests (open-telemetry/donation-openinference, commit 6cdd644d) so its
genuinely-recorded ``inference_conformance.yaml`` cassette replays unchanged.
"""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from crewai import LLM, Agent

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = ("chat",)
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # CrewAI's LLM callback event does not carry a remote server address,
    # and its `model` field is the requested model, not the model name the
    # provider returned, so gen_ai.response.model cannot be populated.
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
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
        env_override = {
            # Keep CrewAI's own telemetry/tracing from phoning home
            # (telemetry.crewai.com) during playback and recording.
            "CREWAI_DISABLE_TELEMETRY": "true",
            "CREWAI_TRACING_ENABLED": "false",
        }
        if not os.getenv("OPENAI_API_KEY"):
            env_override["OPENAI_API_KEY"] = "test_openai_api_key"
        with mock.patch.dict(os.environ, env_override):
            with instrument(
                CrewAIInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                semconv="gen_ai_latest_experimental",
                content_capture="SPAN_ONLY",
            ):
                agent = Agent(
                    role="Helpful Assistant",
                    goal="Answer questions clearly and concisely",
                    backstory="You are a helpful assistant.",
                    allow_delegation=False,
                    llm=LLM(model="gpt-4.1-nano", temperature=0),
                )

                with vcr.use_cassette("inference_conformance.yaml"):
                    agent.kickoff("What is 2+2?")
