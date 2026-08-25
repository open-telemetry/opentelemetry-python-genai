# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Portkey AI chat with tool calls."""

from __future__ import annotations

import os
from typing import Any

from portkey_ai import Portkey

from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class ToolCallingScenario(Scenario):
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
        with instrument(
            PortkeyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("tool_calling_conformance.yaml"):
                client = Portkey(
                    api_key=os.environ.get(
                        "PORTKEY_API_KEY", "test_portkey_api_key"
                    ),
                    provider="google",
                    Authorization=os.environ.get(
                        "GEMINI_API_KEY", "test_gemini_api_key"
                    ),
                )
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string"},
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ]
                messages: list[dict[str, Any]] = [
                    {
                        "role": "user",
                        "content": "What is the weather in SF?",
                    }
                ]
                first = client.chat.completions.create(
                    messages=messages,
                    model="gemini-3.5-flash",
                    tools=tools,
                    stream=False,
                )
                assistant_message = first.choices[0].message
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in (assistant_message.tool_calls or [])
                ]
                messages.append(
                    {
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": tool_calls,
                    }
                )
                for tc in assistant_message.tool_calls or []:
                    messages.append(
                        {
                            "role": "tool",
                            "content": "65 degrees and sunny",
                            "tool_call_id": tc.id,
                        }
                    )
                client.chat.completions.create(
                    messages=messages,
                    model="gemini-3.5-flash",
                    stream=False,
                )
