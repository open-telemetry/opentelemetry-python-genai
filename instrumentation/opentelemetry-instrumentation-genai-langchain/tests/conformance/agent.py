# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: LangChain agent via create_agent."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    LiveCheckReport,
    Scenario,
)
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
    agent_name: str | None = None
    expected_spans = {"invoke_agent": 1, "chat": 3, "execute_tool": 2}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # langchain can't populate server.address on chat spans.
    # invoke_agent provider is unknown at span creation; ls_provider is only
    # available on the chat model callback, not the chain callback.
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
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=100,
                    top_p=0.9,
                    seed=100,
                )
                metadata = {"session_id": "test-session-conformance"}
                if self.agent_name is not None:
                    metadata["agent_name"] = self.agent_name
                agent = create_agent(llm, tools=[multiply, add]).with_config(
                    {"metadata": metadata}
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
        agent_spans = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            if any(
                attr["name"] == "gen_ai.operation.name"
                and attr["value"] == "invoke_agent"
                for attr in entry["span"]["attributes"]
            )
        ]
        assert len(agent_spans) == 1
        agent_span = agent_spans[0]
        agent_attributes = {
            attr["name"]: attr["value"] for attr in agent_span["attributes"]
        }
        expected_span_name = (
            f"invoke_agent {self.agent_name}"
            if self.agent_name is not None
            else "invoke_agent"
        )
        assert agent_span["name"] == expected_span_name
        if self.agent_name is not None:
            assert agent_attributes["gen_ai.agent.name"] == self.agent_name
        else:
            assert "gen_ai.agent.name" not in agent_attributes


class NamedAgentScenario(AgentScenario):
    agent_name = "math_agent"
