# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It is sunny in {city}."


async def main() -> None:
    agent = FunctionAgent(
        name="assistant",
        llm=OpenAI(model="gpt-4o-mini"),
        tools=[FunctionTool.from_defaults(get_weather)],
        streaming=False,
    )
    response = await agent.run(
        user_msg="Use get_weather to check the weather in Paris."
    )
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
