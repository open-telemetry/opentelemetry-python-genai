# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a CrewAI agent with a custom completion hook."""

from crewai import Agent

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.util.genai.completion_hook import CompletionHook
from opentelemetry.util.genai.types import (
    InputMessage,
    MessagePart,
    OutputMessage,
    ToolDefinition,
)


class PrintCompletionHook(CompletionHook):
    """Print content after each CrewAI LLM call."""

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
        print(f"inputs: {inputs}")
        print(f"outputs: {outputs}")


CrewAIInstrumentor().instrument(completion_hook=PrintCompletionHook())

agent = Agent(
    role="Assistant",
    goal="Answer questions concisely",
    backstory="You are a helpful assistant.",
)
print(agent.kickoff("What is OpenTelemetry?"))
