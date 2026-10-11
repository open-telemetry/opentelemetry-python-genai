# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: Google GenAI server-side tool calls."""

from __future__ import annotations

from google.genai import Client

client = Client(api_key="test_google_genai_api_key", vertexai=False)
client.interactions.create(
    model="gemini-2.5-flash",
    input="Search for OpenTelemetry.",
    tools=[{"type": "google_search"}],
)
