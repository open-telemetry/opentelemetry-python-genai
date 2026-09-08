# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: basic agent run for Agno."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse
from agno.tools.function import Function, FunctionCall


class MockModel(Model):
    """Dummy model implementation for testing without provider dependencies."""

    def _parse_provider_response(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _parse_provider_response_delta(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        pass


def sample_tool(x: int) -> int:
    """Double a number."""
    return x * 2


func = Function.from_callable(sample_tool)
func_call = FunctionCall(
    function=func,
    arguments={"x": 5},
    call_id="call-conformance",
)
tool_result = func_call.execute()

agent = Agent(
    name="test-conformance-agent",
    model=MockModel(id="mock-model"),
    session_id="session-conformance",
    tools=[sample_tool],
)
mock_output = ModelResponse(content=f"Double 5 is {tool_result}")
with (
    patch.object(Agent, "run", wraps=agent.run),
    patch("agno.models.base.Model.response", return_value=mock_output),
):
    agent.run(f"Calculate double 5. Tool result: {tool_result}")
