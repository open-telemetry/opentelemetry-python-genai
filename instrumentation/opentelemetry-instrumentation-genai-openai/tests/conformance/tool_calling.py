# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai-v2 chat completion with tool calls."""

import json
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
WEATHER_TOOL_PROMPT = [
    {"role": "system", "content": "You're a helpful assistant."},
    {
        "role": "user",
        "content": "What's the weather in Seattle and San Francisco today?",
    },
]
WEATHER_BY_LOCATION: dict[str, str] = {
    "Seattle, WA": "50 degrees and raining",
    "San Francisco, CA": "70 degrees and sunny",
}


def _get_current_weather_tool_definition() -> dict[str, Any]:
    return {
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
                "additionalProperties": False,
            },
        },
    }


def _execute_weather_tool(arguments: str) -> str:
    try:
        location = json.loads(arguments).get("location", "Seattle, WA")
    except Exception:
        location = "Seattle, WA"
    return WEATHER_BY_LOCATION.get(location, "50 degrees and raining")


client = OpenAI()
messages: list[Any] = list(WEATHER_TOOL_PROMPT)

first = client.chat.completions.create(
    messages=messages,
    model=DEFAULT_MODEL,
    tool_choice="auto",
    tools=[_get_current_weather_tool_definition()],
)

assistant_message = first.choices[0].message
assert assistant_message.tool_calls
messages.append(assistant_message.model_dump(exclude_none=True))
for tc in assistant_message.tool_calls or []:
    messages.append(
        {
            "role": "tool",
            "content": _execute_weather_tool(tc.function.arguments),
            "tool_call_id": tc.id,
        }
    )

client.chat.completions.create(
    messages=messages,
    model=DEFAULT_MODEL,
)
