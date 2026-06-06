# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Zero-code OpenAI Agents example."""

from __future__ import annotations

from agents import Agent, Runner, function_tool
from dotenv import load_dotenv


@function_tool
def get_weather(city: str) -> str:
    """Return a canned weather response for the requested city."""

    return f"The forecast for {city} is sunny with pleasant temperatures."


def main() -> None:
    load_dotenv()
    weather_specialist = Agent(
        name="weather_specialist",
        instructions=(
            "You answer weather questions. Always call the get_weather tool "
            "for the requested city, then summarize the result in one short "
            "sentence with a packing suggestion."
        ),
        tools=[get_weather],
        model="gpt-4o-mini",
    )
    triage_agent = Agent(
        name="triage",
        instructions=(
            "You are a triage agent. If the user asks about weather, "
            "hand off to weather_specialist. Otherwise answer briefly yourself."
        ),
        handoffs=[weather_specialist],
        model="gpt-4o-mini",
    )

    result = Runner.run_sync(
        triage_agent,
        "I'm visiting Barcelona this weekend. How should I pack?",
    )

    print("Agent response:")
    print(result.final_output)


if __name__ == "__main__":
    main()
