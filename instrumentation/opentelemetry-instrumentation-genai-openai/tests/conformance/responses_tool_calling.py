# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API tool calling."""

import json
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
WEATHER_PROMPT = "What's the weather in Seattle and San Francisco today?"
WEATHER_BY_LOCATION: dict[str, dict[str, Any]] = {
    "Seattle, WA": {
        "location": "Seattle, WA",
        "temperature": 58,
        "unit": "F",
        "conditions": "rain",
    },
    "San Francisco, CA": {
        "location": "San Francisco, CA",
        "temperature": 65,
        "unit": "F",
        "conditions": "fog",
    },
}


def _get_current_weather_tool_definition() -> dict[str, Any]:
    """Responses API tools are flat, unlike the nested Chat Completions shape."""
    return {
        "type": "function",
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
        "strict": True,
    }


def _execute_weather_tool(arguments: str) -> str:
    location = json.loads(arguments).get("location", "Seattle, WA")
    return json.dumps(
        WEATHER_BY_LOCATION.get(location, WEATHER_BY_LOCATION["Seattle, WA"])
    )


client = OpenAI()
tools = [_get_current_weather_tool_definition()]

first = client.responses.create(
    model=DEFAULT_MODEL,
    input=WEATHER_PROMPT,
    tool_choice="auto",
    tools=tools,
)

# Replay the model's own output items, then answer each call.
conversation: list[Any] = [
    item.model_dump(exclude_none=True) for item in first.output
]
conversation += [
    {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": _execute_weather_tool(item.arguments),
    }
    for item in first.output
    if item.type == "function_call"
]

client.responses.create(
    model=DEFAULT_MODEL,
    input=conversation,
    tool_choice="auto",
    tools=tools,
)
