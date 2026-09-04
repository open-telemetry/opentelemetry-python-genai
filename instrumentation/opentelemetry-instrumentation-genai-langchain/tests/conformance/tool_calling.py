# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI

TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. Boston, MA",
                },
            },
            "required": ["location"],
        },
    },
}

model = ChatOpenAI(model="gpt-4o-mini", max_tokens=100, temperature=0.5)
with_tools = model.bind_tools([TOOL])

messages = [
    ("system", "You are a helpful assistant."),
    ("human", "What's the weather in Seattle today?"),
]

answer = with_tools.invoke(messages)
assert answer.tool_calls
messages.append(answer)
for call in answer.tool_calls:
    messages.append(
        ToolMessage(
            content=f"70 degrees and sunny in {call['args']['location']}",
            tool_call_id=call["id"],
        )
    )

with_tools.invoke(messages)
