# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a triage agent handing off to a specialist with a tool."""

from agents import Agent, ModelSettings, RunConfig, Runner, function_tool

DEFAULT_MODEL = "gpt-4o-mini"
MODEL_SETTINGS = ModelSettings(
    max_tokens=100,
    temperature=0.5,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.2,
)


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather forecast for the requested city."""
    return f"The forecast for {city} is sunny with a high of 24C."


weather_specialist = Agent(
    name="weather_specialist",
    instructions=(
        "You answer weather questions. Always call the get_weather tool for "
        "the requested city, then summarize the result in one short sentence "
        "with a packing suggestion."
    ),
    tools=[get_weather],
    model=DEFAULT_MODEL,
    model_settings=MODEL_SETTINGS,
)

triage = Agent(
    name="triage",
    instructions=(
        "You are a triage agent. If the user asks about weather, hand off to "
        "weather_specialist. Otherwise answer briefly yourself."
    ),
    handoffs=[weather_specialist],
    model=DEFAULT_MODEL,
    model_settings=MODEL_SETTINGS,
)

Runner.run_sync(
    triage,
    "I'm visiting Barcelona this weekend. Should I pack a jacket?",
    run_config=RunConfig(workflow_name="conformance_workflow"),
)
