# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location."""
    return f"70 degrees and sunny in {location}"


agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=100),
    tools=[get_current_weather],
    system_prompt="You are a helpful assistant.",
    name="weather_assistant",
)

agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in Seattle today?"}
        ]
    }
)
