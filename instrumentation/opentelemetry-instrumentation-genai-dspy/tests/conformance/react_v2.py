# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic ReActV2 agent run with tool execution for DSPy."""

from __future__ import annotations

import dspy
from dspy.adapters.types.tool import ToolCalls


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


react_v2 = dspy.ReActV2("question -> answer", tools=[add])
react_v2.react = MockSyncPredictV2()
react_v2(question="What is 2 + 2?")
