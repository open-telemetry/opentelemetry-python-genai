# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: Google GenAI server-side tool calls."""

from __future__ import annotations

from google.genai import Client

client = Client(api_key="test_google_genai_api_key", vertexai=False)
client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Run Python to print one.",
    config={"tools": [{"code_execution": {}}]},
)
