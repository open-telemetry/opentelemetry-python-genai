# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a chat turn where the model returns a tool call.

Asserts the tool call round-trips onto the output message's ``tool_call``
part; weaver validates the part's shape.
"""

from __future__ import annotations

import json
from typing import Any

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ._known_gaps import MISSING_RESPONSE_ID, MISSING_SERVER_ADDRESS

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }
            },
            "required": ["location"],
        },
    },
}


class ToolCallingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (MISSING_RESPONSE_ID, MISSING_SERVER_ADDRESS)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            HaystackInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("tool_calling_conformance.yaml"):
                OpenAIChatGenerator(model="gpt-4o").run(
                    messages=[
                        ChatMessage.from_user("What is the weather in Berlin")
                    ],
                    generation_kwargs={"tools": [_WEATHER_TOOL]},
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        chat_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert chat_spans, "no chat span emitted"
        output_part_types = {
            t
            for span in chat_spans
            for t in _part_types(_attr(span, "gen_ai.output.messages"))
        }
        assert "tool_call" in output_part_types, (
            f"expected a tool_call part on an output message, saw {output_part_types}"
        )


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None


def _part_types(messages_json: str | None) -> list[str]:
    messages = json.loads(messages_json) if messages_json else []
    return [part["type"] for message in messages for part in message["parts"]]
