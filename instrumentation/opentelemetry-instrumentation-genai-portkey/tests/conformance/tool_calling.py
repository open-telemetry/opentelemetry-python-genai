# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from portkey_ai import Portkey

client = Portkey(
    api_key="test_portkey_api_key",
    base_url=f"{os.environ['MOCK_SERVER_URL']}/v1",
    provider="openai",
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
            },
        },
    }
]

messages = [
    {"role": "user", "content": "What's the weather in Seattle today?"}
]

first = client.chat.completions.create(
    messages=messages,
    model="gpt-4o-mini",
    tools=tools,
    stream=False,
)

assistant_message = first.choices[0].message
assert assistant_message.tool_calls
tool_calls = [
    {
        "id": tc.id,
        "type": tc.type,
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments,
        },
    }
    for tc in (assistant_message.tool_calls or [])
]
messages.append(
    {
        "role": "assistant",
        "content": assistant_message.content,
        "tool_calls": tool_calls,
    }
)
for tc in assistant_message.tool_calls or []:
    messages.append(
        {
            "role": "tool",
            "content": "70 degrees and sunny in Seattle",
            "tool_call_id": tc.id,
        }
    )

client.chat.completions.create(
    messages=messages,
    model="gpt-4o-mini",
    stream=False,
)
