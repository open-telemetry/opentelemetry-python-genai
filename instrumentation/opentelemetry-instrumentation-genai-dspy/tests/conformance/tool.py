# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: standalone tool execution for DSPy."""

from __future__ import annotations

import dspy


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


tool = dspy.Tool(multiply, name="multiply", desc="Multiply two numbers.")
tool(a=3, b=4)
