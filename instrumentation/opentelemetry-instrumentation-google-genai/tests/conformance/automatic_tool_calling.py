# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai tool execution."""

from google.genai import Client, types


def get_weather(location: str) -> str:
    """Get weather for location"""
    return "sunny"


client = Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the weather in Boston?",
    config={"tools": [get_weather]},
)

contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the weather in Boston?")],
    ),
    response.candidates[0].content,
    types.Content(
        role="user",
        parts=[
            types.Part.from_function_response(
                name="get_weather",
                response={"result": "sunny"},
            )
        ],
    ),
]

client.models.generate_content(
    model="gemini-2.5-flash",
    contents=contents,
    config={"tools": [get_weather]},
)
