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
from opentelemetry.test_util_genai.conformance import Scenario
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
