# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic ReAct agent run with tool execution for DSPy."""

from __future__ import annotations

from typing import Any

import dspy

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


class MockSyncPredict:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, **kwargs: object) -> object:
        self.count += 1

        class Pred:
            next_thought = "Need to calculate 2 + 2"
            next_tool_name = "add"
            next_tool_args = {"x": 2, "y": 2}

        class FinishPred:
            next_thought = "Done calculation"
            next_tool_name = "finish"
            next_tool_args = {}

        return Pred() if self.count == 1 else FinishPred()


class MockExtract:
    def __call__(self, **kwargs: object) -> dict[str, str]:
        return {"answer": "4"}


class ReActScenario(Scenario):
    expected_spans = {"invoke_agent": 1, "execute_tool": 2}
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
            tool = dspy.Tool(add, name="add", desc="Add two numbers.")
            react = dspy.ReAct("question -> answer", tools=[tool])
            react.react = MockSyncPredict()
            react.extract = MockExtract()

            react(question="What is 2 + 2?")
