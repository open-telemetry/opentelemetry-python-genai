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
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
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
    # before producing the final answer: 3 chat turns and 2 tool calls.
    # Assistant internally runs its Memory sub-agent (also an Agent), which
    # accounts for the second invoke_agent span.
    expected_spans = {"invoke_agent": 2, "chat": 3, "execute_tool": 2}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # qwen-agent does not expose the resolved model service endpoint, so
    # server.address cannot be populated on chat spans.
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
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

        # Weaver validates the shape of each message part, but not that the
        # tool call actually round-tripped. Assert the model's tool_call
        # landed on an output message and the tool result was fed back on an
        # input message (across the two chat turns).
        chat_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert chat_spans, "no chat span emitted"

        output_part_types = {
            part_type
            for span in chat_spans
            for part_type in _part_types(_attr(span, "gen_ai.output.messages"))
        }
        input_part_types = {
            part_type
            for span in chat_spans
            for part_type in _part_types(_attr(span, "gen_ai.input.messages"))
        }
        assert "tool_call" in output_part_types, (
            "expected a tool_call part on an output message, saw"
            f" {output_part_types}"
        )
        assert "tool_call_response" in input_part_types, (
            "expected a tool_call_response part on an input message, saw"
            f" {input_part_types}"
        )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    # gen_ai.{input,output}.messages is a JSON string of
    # [{"role": ..., "parts": [{"type": ..., ...}]}].
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]
