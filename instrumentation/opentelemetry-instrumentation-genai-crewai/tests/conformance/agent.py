# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario for a CrewAI agent executing a tool."""

from __future__ import annotations

from typing import Any

from crewai import Agent, BaseLLM, Task
from crewai.tools import BaseTool

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _AddTool(BaseTool):
    name: str = "add"
    description: str = "Add two integers"

    def _run(self, left: int, right: int) -> int:
        return left + right


class _ScriptedLLM(BaseLLM):
    def __init__(self) -> None:
        super().__init__(model="scripted-model")
        self._calls = 0

    def call(
        self,
        messages: Any,
        tools: Any = None,
        callbacks: Any = None,
        available_functions: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
    ) -> str:
        del (
            messages,
            tools,
            callbacks,
            available_functions,
            from_task,
            from_agent,
            response_model,
        )
        self._calls += 1
        if self._calls == 1:
            return (
                "Thought: I should add.\n"
                "Action: add\n"
                'Action Input: {"left": 2, "right": 3}'
            )
        return "Thought: done.\nFinal Answer: 5"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False


class AgentScenario(Scenario):
    expected_spans = {"invoke_agent": 1, "execute_tool": 1}
    expected_metrics = (
        "gen_ai.invoke_agent.duration",
        "gen_ai.execute_tool.duration",
    )
    # The tool call id is only visible in CrewAI's native function-calling
    # executor, which is not patched yet; the ReAct path has no id at all.
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
        del vcr
        with instrument(
            CrewAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
            extra_env={"CREWAI_DISABLE_TELEMETRY": "true"},
        ):
            tool = _AddTool()
            agent = Agent(
                role="Calculator",
                goal="Calculate accurately",
                backstory="An offline conformance agent",
                llm=_ScriptedLLM(),
                tools=[tool],
                verbose=False,
            )
            task = Task(
                description="Add two and three",
                expected_output="The number five",
                agent=agent,
            )
            agent.execute_task(task)
