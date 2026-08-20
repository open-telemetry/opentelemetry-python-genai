# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for the agent operations (``invoke_agent`` with a nested
``execute_tool``)."""

from __future__ import annotations

import pathlib
from typing import Any

from smolagents import OpenAIModel, ToolCallingAgent

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class AgentScenario(Scenario):
    # ToolCallingAgent runs one step, which calls final_answer: one
    # execute_tool span under one invoke_agent span.
    expected_spans = {"invoke_agent": 1, "execute_tool": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        from PIL import Image

        image_path = (
            pathlib.Path(__file__).parent.parent / "fixtures" / "img.png"
        )
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("agent_with_image.yaml"):
                model = OpenAIModel(
                    model_id="gpt-4o",
                    api_key="test_openai_api_key",
                    api_base="https://api.openai.com/v1",
                )
                agent = ToolCallingAgent(tools=[], model=model, max_steps=3)
                agent.run(
                    "Describe what you see in this image briefly.",
                    images=[Image.open(image_path)],
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        agent_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.agent.name"
        }
        assert "ToolCallingAgent" in agent_names, (
            f"expected the agent name on the invoke_agent span, saw {agent_names}"
        )
        tool_call_ids = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.tool.call.id"
        }
        assert tool_call_ids, (
            "expected the provider's tool call id on the execute_tool span"
        )
