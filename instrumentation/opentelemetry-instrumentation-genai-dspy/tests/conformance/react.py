# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic ReAct agent run with tool execution for DSPy."""

from __future__ import annotations

import dspy


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


tool = dspy.Tool(add, name="add", desc="Add two numbers.")
react = dspy.ReAct("question -> answer", tools=[tool])
react.react = MockSyncPredict()
react.extract = MockExtract()
react(question="What is 2 + 2?")
