# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain two-node LangGraph workflow (researcher → summariser)."""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict
from unittest import mock

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

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


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    research: str


class WorkflowScenario(Scenario):
    expected_spans = ("invoke_workflow", "chat", "chat")
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
                    max_tokens=200,
                    seed=42,
                )

                def researcher(state: GraphState) -> dict:
                    response = llm.invoke(
                        [
                            SystemMessage(
                                content="You are a research assistant. Provide 2-3 factual sentences."
                            ),
                            HumanMessage(
                                content=state["messages"][-1].content
                            ),
                        ]
                    )
                    return {
                        "research": response.content,
                        "messages": [response],
                    }

                def summariser(state: GraphState) -> dict:
                    response = llm.invoke(
                        [
                            SystemMessage(
                                content="You are an expert summariser. Condense the text below into one clear sentence."
                            ),
                            HumanMessage(content=state["research"]),
                        ]
                    )
                    return {"messages": [response]}

                builder = StateGraph(GraphState)
                builder.add_node("researcher", researcher)
                builder.add_node("summariser", summariser)
                builder.add_edge(START, "researcher")
                builder.add_edge("researcher", "summariser")
                builder.add_edge("summariser", END)
                graph = builder.compile()

                with vcr.use_cassette("workflow_conformance.yaml"):
                    graph.invoke(
                        {
                            "messages": [
                                HumanMessage(
                                    content="What is the capital of France?"
                                )
                            ],
                            "research": "",
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
        assert "invoke_workflow" in operations, (
            f"Expected an invoke_workflow span; saw operations {operations}"
        )
        assert operations.count("chat") >= 2, (
            "Two-node workflow exercises two chat completions "
            f"(researcher and summariser); saw {operations}"
        )
