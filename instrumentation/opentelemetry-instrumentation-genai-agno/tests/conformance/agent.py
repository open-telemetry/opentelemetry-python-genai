# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic agent run for Agno."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agno.agent import Agent
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall
from tests.mock_model import MockModel

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class AgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1, "execute_tool": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):

            def sample_tool(x: int) -> int:
                """Double a number."""
                return x * 2

            func = Function.from_callable(sample_tool)
            func_call = FunctionCall(
                function=func,
                arguments={"x": 5},
                call_id="call-conformance",
            )
            func_call.execute()

            agent = Agent(
                name="test-conformance-agent",
                model=MockModel(id="mock-model"),
                session_id="session-conformance",
                tools=[sample_tool],
            )
            mock_output = ModelResponse(content="Conformance Hello back!")
            with (
                patch.object(Agent, "run", wraps=agent.run),
                patch(
                    "agno.models.base.Model.response", return_value=mock_output
                ),
            ):
                agent.run("hello conformance world")
