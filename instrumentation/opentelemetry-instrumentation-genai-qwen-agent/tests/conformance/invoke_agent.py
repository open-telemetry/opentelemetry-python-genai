# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a tool-calling qwen-agent Assistant run."""

from __future__ import annotations

import json
from typing import Any

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    LiveCheckReport,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


@register_tool("get_current_weather_test", allow_overwrite=True)
class _GetCurrentWeatherTool(BaseTool):
    description = "Get the current weather for a given city."
    parameters = [
        {
            "name": "city",
            "type": "string",
            "description": "The city name to get weather for.",
            "required": True,
        }
    ]

    def call(self, params: Any, **kwargs: Any) -> str:
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                params = {"city": params}
        city = (
            params.get("city", "unknown")
            if isinstance(params, dict)
            else "unknown"
        )
        return f"The weather in {city} is sunny and 22 degrees Celsius."


class InvokeAgentScenario(Scenario):
    # The recorded run has the model requesting the weather tool twice
    # before producing the final answer. Assistant internally runs its
    # Memory sub-agent (also an Agent), which accounts for the second
    # invoke_agent span. LLM call (chat) spans are left to the underlying
    # model client library's instrumentation and are not emitted here.
    expected_spans = {"invoke_agent": 2, "execute_tool": 2}
    expected_metrics = (
        "gen_ai.invoke_agent.duration",
        "gen_ai.invoke_agent.inference_calls",
        "gen_ai.invoke_agent.tool_calls",
        "gen_ai.execute_tool.duration",
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
            QwenAgentInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            bot = Assistant(
                llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
                name="WeatherAgent",
                function_list=["get_current_weather_test"],
            )
            with vcr.use_cassette("invoke_agent.yaml"):
                list(
                    bot.run(
                        [
                            {
                                "role": "user",
                                "content": (
                                    "What is the weather in Beijing right now?"
                                ),
                            }
                        ]
                    )
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        metrics = {
            entry["metric"]["name"]: entry["metric"]
            for entry in report["samples"]
            if "metric" in entry
        }
        # The outer Assistant takes three model turns (two of them request the
        # tool) and calls the tool twice. Its Memory sub-agent makes no model
        # or tool call of its own, so both of its counts are zero.
        expected = {
            "gen_ai.invoke_agent.inference_calls": {
                "WeatherAgent": 3,
                "Memory": 0,
            },
            "gen_ai.invoke_agent.tool_calls": {
                "WeatherAgent": 2,
                "Memory": 0,
            },
        }
        for name, per_agent in expected.items():
            metric = metrics[name]
            assert metric["instrument"] == "histogram"
            observed: dict[str, tuple[int, float]] = {}
            for point in metric["data_points"]:
                attributes = {
                    attribute["name"]: attribute["value"]
                    for attribute in point["attributes"]
                }
                observed[attributes["gen_ai.agent.name"]] = (
                    point["count"],
                    point["sum"],
                )
            assert observed == {
                agent: (1, total) for agent, total in per_agent.items()
            }
