# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai interactions function tool calling."""

from __future__ import annotations

import os
from typing import Any

from google.genai import Client
from google.genai.interactions import (
    Function,
    FunctionCallStep,
    FunctionResultStep,
    TextContent,
)

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

_MODEL = "gemini-2.5-flash"
_PROMPT = "What is the weather in Paris?"
_WEATHER_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
}
_FORECAST_PARAMS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "days": {"type": "integer"},
    },
    "required": ["city"],
}
_TOOL_RESULT_TEXT = "22C and sunny"
_FIRST_INTERACTION_ID = (
    "v1_ChdUb29sQ2FsbENvbmZvcm1hbmNlVHVybjESF1Rvb2xDYWxsQ29uZm9ybWFuY2VUdXJuMQ"
)
_CALL_ID = "call_get_weather_1"
_FINAL_ANSWER = "The weather in Paris is 22C and sunny."

_EXPECTED_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": _WEATHER_PARAMS,
    },
    {
        "type": "function",
        "name": "get_forecast",
        "description": "Get the multi-day forecast for a city.",
        "parameters": _FORECAST_PARAMS,
    },
]


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


class InteractionsToolCallingScenario(Scenario):
    expected_spans = {"chat": 2}
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
        api_key = (
            os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
            or "test_google_genai_api_key"
        )
        tools = [
            Function(
                name="get_weather",
                description="Get the current weather for a city.",
                parameters=_WEATHER_PARAMS,
            ),
            Function(
                name="get_forecast",
                description="Get the multi-day forecast for a city.",
                parameters=_FORECAST_PARAMS,
            ),
        ]
        with instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "interactions_tool_calling_conformance.yaml"
            ):
                client = Client(api_key=api_key, vertexai=False)
                first_response = client.interactions.create(
                    model=_MODEL,
                    input=_PROMPT,
                    tools=tools,
                )
                call_step = next(
                    step
                    for step in (first_response.steps or [])
                    if isinstance(step, FunctionCallStep)
                )
                client.interactions.create(
                    model=_MODEL,
                    previous_interaction_id=first_response.id,
                    input=[
                        FunctionResultStep(
                            call_id=call_step.id,
                            name=call_step.name,
                            result=[TextContent(text=_TOOL_RESULT_TEXT)],
                        )
                    ],
                    tools=tools,
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert len(spans) == 2
        first_span, second_span = spans

        assert (
            _attr(first_span, "gen_ai.tool.definitions")
            == _EXPECTED_TOOL_DEFINITIONS
        )
        assert _attr(first_span, "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": [{"type": "text", "content": _PROMPT}],
            }
        ]
        assert _attr(first_span, "gen_ai.output.messages") == [
            {
                "role": "assistant",
                "parts": [
                    {
                        "type": "tool_call",
                        "id": _CALL_ID,
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    }
                ],
            }
        ]
        assert _attr(first_span, "gen_ai.response.finish_reasons") == [
            "tool_calls"
        ]

        assert (
            _attr(second_span, "gen_ai.tool.definitions")
            == _EXPECTED_TOOL_DEFINITIONS
        )
        assert (
            _attr(second_span, "gen_ai.request.previous_response.id")
            == _FIRST_INTERACTION_ID
        )
        assert _attr(second_span, "gen_ai.input.messages") == [
            {
                "role": "user",
                "parts": [
                    {
                        "type": "tool_call_response",
                        "id": _CALL_ID,
                        "response": [
                            {"type": "text", "text": _TOOL_RESULT_TEXT}
                        ],
                    }
                ],
            }
        ]
        assert _attr(second_span, "gen_ai.output.messages") == [
            {
                "role": "assistant",
                "parts": [{"type": "text", "content": _FINAL_ANSWER}],
            }
        ]
        assert _attr(second_span, "gen_ai.response.finish_reasons") == ["stop"]
