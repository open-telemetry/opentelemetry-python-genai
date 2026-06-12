# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: triage agent hands off to a specialist that uses a tool.

Exercises the three common agent shapes in a single ``Runner.run``:

- Basic agent invocation (``invoke_agent`` + ``chat`` from the Responses API).
- Multi-agent handoff (a second ``invoke_agent`` after the triage step).
- Function tool execution (``execute_tool``) inside the specialist agent.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest import mock

from agents import Agent, Runner, function_tool

from opentelemetry.instrumentation.genai.openai_agents import (
    OpenAIAgentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

DEFAULT_MODEL = "gpt-4o-mini"


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather forecast for the requested city."""

    return f"The forecast for {city} is sunny with a high of 24C."


def _build_triage_agent() -> Agent:
    weather_specialist = Agent(
        name="weather_specialist",
        instructions=(
            "You answer weather questions. Always call the get_weather tool "
            "for the requested city, then summarize the result in one short "
            "sentence with a packing suggestion."
        ),
        tools=[get_weather],
        model=DEFAULT_MODEL,
    )
    return Agent(
        name="triage",
        instructions=(
            "You are a triage agent. If the user asks about weather, "
            "hand off to weather_specialist. Otherwise answer briefly yourself."
        ),
        handoffs=[weather_specialist],
        model=DEFAULT_MODEL,
    )


class OrchestrationScenario(Scenario):
    expected_spans = (
        "invoke_agent",
        "chat",
        "execute_tool",
    )
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
        key_override = (
            {}
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                OpenAIAgentsInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                semconv="gen_ai_latest_experimental",
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("orchestration_conformance.yaml"):
                    triage = _build_triage_agent()
                    asyncio.run(
                        Runner.run(
                            triage,
                            "I'm visiting Barcelona this weekend. "
                            "Should I pack a jacket?",
                        )
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
        agent_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.agent.name"
        }
        assert operations.count("invoke_agent") >= 2, (
            "Orchestration involves a triage agent handing off to a specialist; "
            f"expected at least two invoke_agent spans, saw {operations}"
        )
        assert operations.count("chat") >= 2, (
            "Each agent issues at least one Responses-API call; "
            f"expected at least two chat spans, saw {operations}"
        )
        assert operations.count("execute_tool") >= 1, (
            "Specialist agent calls the get_weather function tool; "
            f"expected at least one execute_tool span, saw {operations}"
        )
        assert len(agent_names) >= 2, (
            "Triage and specialist must each surface their own gen_ai.agent.name; "
            f"saw {sorted(agent_names)}"
        )
