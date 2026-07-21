# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a DashScope-backed AgentScope v2 agent uses a tool.

Exercises the three agent shapes a single ``Agent.reply`` produces when the
model decides to call a function tool:

- Basic agent invocation (``invoke_agent``).
- The chat completions the agent issues to its model (``chat``) — one to
  request the tool call and one to summarize the tool result.
- Function tool execution (``execute_tool``).

The agent runs with ``PermissionMode.BYPASS`` so the recorded ReAct loop
executes the tool without a human-in-the-loop confirmation prompt.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest import mock

from agentscope.agent import Agent
from agentscope.credential import DashScopeCredential
from agentscope.message import TextBlock, UserMsg
from agentscope.model import DashScopeChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.tool import FunctionTool, Toolkit, ToolResponse

from opentelemetry.instrumentation.genai.agentscope import (
    AgentScopeInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

DEFAULT_MODEL = "qwen-plus"


def get_weather(city: str) -> ToolResponse:
    """Look up the current weather for a city."""
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=f"The weather in {city} is sunny with a high of 24C.",
            )
        ]
    )


def _build_model() -> DashScopeChatModel:
    return DashScopeChatModel(
        credential=DashScopeCredential(
            api_key=os.environ["DASHSCOPE_API_KEY"]
        ),
        model=DEFAULT_MODEL,
        parameters=DashScopeChatModel.Parameters(
            max_tokens=256,
            thinking_enable=False,
        ),
        stream=False,
        max_retries=0,
    )


def _build_agent() -> Agent:
    return Agent(
        name="weather_agent",
        system_prompt=(
            "You answer weather questions. Always call the get_weather tool "
            "for the requested city, then reply in one short sentence."
        ),
        model=_build_model(),
        toolkit=Toolkit(tools=[FunctionTool(get_weather)]),
        state=AgentState(
            permission_context=PermissionContext(mode=PermissionMode.BYPASS)
        ),
    )


class OrchestrationScenario(Scenario):
    expected_spans = {
        "invoke_agent": 1,
        # The agent issues one chat to request the tool call and one to
        # summarize the tool result.
        "chat": 2,
        "execute_tool": 1,
    }
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
        key_override = (
            {}
            if os.getenv("DASHSCOPE_API_KEY")
            else {"DASHSCOPE_API_KEY": "test_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                AgentScopeInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette("orchestration_conformance.yaml"):
                    agent = _build_agent()
                    asyncio.run(
                        agent.reply(
                            UserMsg(
                                name="user",
                                content="What is the weather in Hangzhou?",
                            )
                        )
                    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        # Presence of the invoke_agent, chat, and execute_tool operations is
        # already enforced by ``expected_spans``; only the agent-name attribute
        # needs a scenario-specific check.
        agent_names = {
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.agent.name"
        }
        assert agent_names, (
            "invoke_agent span must carry gen_ai.agent.name; "
            f"saw {sorted(agent_names)}"
        )
