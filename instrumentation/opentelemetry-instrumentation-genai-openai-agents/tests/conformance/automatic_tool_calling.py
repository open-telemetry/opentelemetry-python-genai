# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: an openai-agents run that calls a tool."""

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool


@function_tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f"70 degrees and sunny in {location}"


agent = Agent(
    name="weather_assistant",
    instructions="You are a helpful assistant.",
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        max_tokens=100,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    ),
    tools=[get_current_weather],
)

Runner.run_sync(
    agent,
    "What's the weather in Seattle today?",
    run_config=RunConfig(workflow_name="conformance_workflow"),
)
