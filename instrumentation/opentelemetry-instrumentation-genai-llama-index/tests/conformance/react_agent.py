# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio

from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI

agent = ReActAgent(
    name="react-agent",
    llm=OpenAI(model="gpt-4o-mini", temperature=0.1),
    streaming=False,
)


async def main() -> None:
    await agent.run("What is two plus two?")


asyncio.run(main())
