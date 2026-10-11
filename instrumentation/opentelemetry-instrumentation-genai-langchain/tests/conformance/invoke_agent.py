# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a named langchain agent."""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=100),
    tools=[],
    system_prompt="You are a helpful assistant.",
    name="weather_assistant",
)
agent.invoke({"messages": [{"role": "user", "content": "Say this is a test"}]})
