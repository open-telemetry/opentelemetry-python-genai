# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: an openai chat completion with a JSON schema answer."""

from openai import OpenAI

SCHEMA = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "temperature": {"type": "integer"},
        "conditions": {"enum": ["sunny", "cloudy", "rainy"]},
    },
    "required": ["location", "temperature", "conditions"],
    "additionalProperties": False,
}

client = OpenAI()
client.chat.completions.create(
    model="gpt-4o-mini",
    max_tokens=100,
    temperature=0.5,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Seattle?"},
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "weather_report",
            "strict": True,
            "schema": SCHEMA,
        },
    },
)
