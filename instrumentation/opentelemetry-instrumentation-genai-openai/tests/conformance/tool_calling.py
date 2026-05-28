# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai-v2 chat completion with tool calls."""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

DEFAULT_MODEL = "gpt-4o-mini"
WEATHER_TOOL_PROMPT = [
    {"role": "system", "content": "You're a helpful assistant."},
    {
        "role": "user",
        "content": "What's the weather in Seattle and San Francisco today?",
    },
]
# Tool outputs are pinned to the recorded cassette's second request body.
WEATHER_BY_LOCATION: dict[str, str] = {
    "Seattle, WA": "50 degrees and raining",
    "San Francisco, CA": "70 degrees and sunny",
}


def _get_current_weather_tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. Boston, MA",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }


def _execute_weather_tool(arguments: str) -> str:
    location = json.loads(arguments)["location"]
    return WEATHER_BY_LOCATION[location]


class ToolCallingScenario(Scenario):
    expected_spans = ("chat",)
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
        with instrument(
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            semconv="gen_ai_latest_experimental",
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("tool_calling_conformance.yaml"):
                client = OpenAI()
                messages: list[Any] = list(WEATHER_TOOL_PROMPT)

                first = client.chat.completions.create(
                    messages=messages,
                    model=DEFAULT_MODEL,
                    tool_choice="auto",
                    tools=[_get_current_weather_tool_definition()],
                )

                assistant_message = first.choices[0].message
                messages.append(
                    assistant_message.model_dump(exclude_none=True)
                )
                for tc in assistant_message.tool_calls or []:
                    messages.append(
                        {
                            "role": "tool",
                            "content": _execute_weather_tool(
                                tc.function.arguments
                            ),
                            "tool_call_id": tc.id,
                        }
                    )

                client.chat.completions.create(
                    messages=messages,
                    model=DEFAULT_MODEL,
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
        assert operations == ["chat", "chat"], (
            "Tool calling exercises two chat completions (initial request and "
            f"follow-up with tool results); saw spans {operations}"
        )
