# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: an Agent run that calls a tool.

Uses a fake, local, deterministic ``ChatGenerator`` (scripted to request a
tool call, then answer) and a real ``Tool`` -- no HTTP interaction, no
cassette needed. Exercises ``invoke_agent`` plus the nested ``chat`` /
``execute_tool`` spans the Agent drives.
"""

from __future__ import annotations

from typing import Any, List

from haystack import component
from haystack.components.agents.agent import Agent
from haystack.dataclasses.chat_message import ChatMessage, ToolCall
from haystack.tools import Tool

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ._known_gaps import MISSING_SERVER_ADDRESS, MISSING_TOOL_CALL_ID


@component
class _ScriptedChatGenerator:
    def __init__(self) -> None:
        self.model = "scripted-model"
        self._call_count = 0

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: Any,  # noqa: ARG002
        tools: Any = None,  # noqa: ARG002
        generation_kwargs: Any = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        self._call_count += 1
        if self._call_count == 1:
            reply = ChatMessage.from_assistant(
                text=None,
                tool_calls=[
                    ToolCall(
                        tool_name="get_weather",
                        arguments={"city": "Berlin"},
                        id="call_1",
                    )
                ],
                meta={
                    "id": "scripted-response-1",
                    "model": "scripted-model",
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )
        else:
            reply = ChatMessage.from_assistant(
                text="It is sunny in Berlin.",
                meta={
                    "id": "scripted-response-2",
                    "model": "scripted-model",
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
                },
            )
        return {"replies": [reply]}


def _get_weather(city: str) -> str:
    return f"sunny in {city}"


class InvokeAgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1, "chat": 2, "execute_tool": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    # _ScriptedChatGenerator is a bare test fake with no SDK client at all,
    # so server.address is never available for it (not just a cold-start
    # timing issue, unlike the real OpenAI-backed scenarios).
    expected_violations = (MISSING_SERVER_ADDRESS, MISSING_TOOL_CALL_ID)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,  # noqa: ARG002 - unused; no HTTP interaction
    ) -> None:
        with instrument(
            HaystackInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            tool = Tool(
                name="get_weather",
                description="Get the weather for a city",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
                function=_get_weather,
            )
            agent = Agent(
                chat_generator=_ScriptedChatGenerator(), tools=[tool]
            )
            agent.run(
                messages=[
                    ChatMessage.from_user("What's the weather in Berlin?")
                ]
            )
