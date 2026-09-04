# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai structured output."""

from google.genai import Client, types
from pydantic import BaseModel


class WeatherReport(BaseModel):
    temperature: float
    conditions: str


client = Client()
client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What's the weather in Seattle?",
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=WeatherReport,
    ),
)
