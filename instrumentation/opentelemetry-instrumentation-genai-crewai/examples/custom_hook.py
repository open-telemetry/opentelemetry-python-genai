# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Instrument CrewAI with a custom completion hook."""

from crewai import Agent, Task

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)


class PrintCompletionHook:
    """Print captured agent inputs and outputs."""

    def on_completion(
        self,
        *,
        inputs: list[InputMessage],
        outputs: list[OutputMessage],
        system_instruction: list[MessagePart],
        tool_definitions: list[ToolDefinition] | None = None,
        span=None,
        log_record=None,
    ) -> None:
        del system_instruction, tool_definitions, span, log_record
        print(f"inputs: {inputs}")
        print(f"outputs: {outputs}")


CrewAIInstrumentor().instrument(completion_hook=PrintCompletionHook())

agent = Agent(
    role="Writer",
    goal="Write concise answers",
    backstory="An example CrewAI agent",
)
task = Task(
    description="Explain OpenTelemetry in one sentence.",
    expected_output="One sentence",
    agent=agent,
)
print(agent.execute_task(task))
