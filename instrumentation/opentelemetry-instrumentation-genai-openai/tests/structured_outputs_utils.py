# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared test definitions for structured outputs (parse) tests."""

from pydantic import BaseModel


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]


STRUCTURED_OUTPUT_PROMPT = [
    {
        "role": "user",
        "content": "Extract the event information from: Team Meeting on 2024-01-15 with Alice and Bob",
    }
]

STRUCTURED_OUTPUT_EXPECTED_INPUT_MESSAGES = [
    {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "content": STRUCTURED_OUTPUT_PROMPT[0]["content"],
            }
        ],
    }
]
