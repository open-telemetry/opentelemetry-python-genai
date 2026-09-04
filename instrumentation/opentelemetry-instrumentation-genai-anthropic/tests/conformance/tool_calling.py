# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic chat with tool calls."""

from anthropic import Anthropic

client = Anthropic()
tools = [
    {
        "name": "get_weather",
        "description": "Get weather by city",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]
messages = [
    {
        "role": "user",
        "content": "What is the weather in SF?",
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=messages,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
)

tool_results = []
for block in response.content:
    if block.type == "tool_use":
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "70 degrees and sunny",
            }
        )

messages.append({"role": "assistant", "content": response.content})
messages.append({"role": "user", "content": tool_results})

client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=messages,
    tools=tools,
)
