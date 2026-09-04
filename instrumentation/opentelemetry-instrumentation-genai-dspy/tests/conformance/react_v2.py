# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic ReActV2 agent run with tool execution for DSPy."""

from __future__ import annotations

from typing import Any

import dspy
from dspy.adapters.types.tool import ToolCalls
from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


class MockSyncPredictV2:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, **kwargs: object) -> object:
        self.count += 1

        class CallPred:
            next_thought = "Add 2 and 2"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_1", name="add", args={"x": 2, "y": 2}
                    )
                ]
            )

        class SubmitPred:
            next_thought = "Submit answer"
            tool_calls = ToolCalls(
                tool_calls=[
                    ToolCalls.ToolCall(
                        id="call_2", name="submit", args={"answer": "4"}
                    )
                ]
            )

        return CallPred() if self.count == 1 else SubmitPred()


class ReActV2Scenario(Scenario):
    expected_spans = {"invoke_agent": 1, "execute_tool": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.tool.call.id",
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
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            react_v2 = dspy.ReActV2("question -> answer", tools=[add])
            react_v2.react = MockSyncPredictV2()
            react_v2(question="What is 2 + 2?")
