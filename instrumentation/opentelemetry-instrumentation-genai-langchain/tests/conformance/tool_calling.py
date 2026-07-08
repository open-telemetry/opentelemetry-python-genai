# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain chat with tool calling via ChatOpenAI."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest import mock

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.util.genai.handler import TelemetryHandler

DEFAULT_MODEL = "gpt-4o-mini"
WEATHER_TOOL_PROMPT = [
    SystemMessage(content="You're a helpful assistant."),
    HumanMessage(
        content="What's the weather in Seattle and San Francisco today?"
    ),
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
    expected_spans = ("chat", "execute_tool")
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # langchain can't populate server.address on chat spans.
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
        key_override = (
            {}
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        tool_handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                llm = ChatOpenAI(
                    model=DEFAULT_MODEL,
                    tool_choice="auto",
                )
                llm_with_tools = llm.bind_tools(
                    [_get_current_weather_tool_definition()]
                )

                messages: list[Any] = list(WEATHER_TOOL_PROMPT)

                with vcr.use_cassette("tool_calling_conformance.yaml"):
                    first_response: AIMessage = llm_with_tools.invoke(messages)
                    messages.append(first_response)

                    for tool_call in first_response.tool_calls:
                        with tool_handler.tool(
                            tool_call["name"],
                            tool_call_id=tool_call["id"],
                            tool_type="function",
                        ) as invocation:
                            result = _execute_weather_tool(
                                json.dumps(tool_call["args"])
                            )
                            invocation.tool_result = result
                        messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call["id"],
                            )
                        )

                    llm_with_tools.invoke(messages)

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        operations = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.operation.name"
        ]
        assert operations == [
            "chat",
            "execute_tool",
            "execute_tool",
            "chat",
        ], (
            "Tool calling exercises two chat completions with two execute_tool "
            "spans in between (one per tool call); saw spans {operations}"
        )
