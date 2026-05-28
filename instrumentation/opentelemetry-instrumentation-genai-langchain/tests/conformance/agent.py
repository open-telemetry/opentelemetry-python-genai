# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain ReAct agent via LangGraph create_react_agent."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


class AgentScenario(Scenario):
    expected_spans = ("invoke_agent", "chat", "chat")
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
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                semconv="gen_ai_latest_experimental",
                content_capture="SPAN_ONLY",
            ):
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=100,
                    top_p=0.9,
                    seed=100,
                )
                agent = create_react_agent(
                    llm, tools=[multiply, add]
                ).with_config(
                    {
                        "metadata": {
                            "agent_name": "math_agent",
                            "session_id": "test-session-conformance",
                        },
                    }
                )
                with vcr.use_cassette("agent_conformance.yaml"):
                    agent.invoke(
                        {
                            "messages": [
                                HumanMessage(content="What is (3 * 4) + 7?")
                            ]
                        }
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
        assert "invoke_agent" in operations, (
            f"Expected an invoke_agent span; saw operations {operations}"
        )
        assert operations.count("chat") >= 2, (
            "ReAct agent exercises at least two chat completions "
            f"(initial tool-call request and follow-up); saw {operations}"
        )
