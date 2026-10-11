# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai chat completion (inference)."""

from google.genai import Client

client = Client()
client.interactions.create(
    model="gemini-2.5-flash",
    input="Hello, how can you help me today?",
)
