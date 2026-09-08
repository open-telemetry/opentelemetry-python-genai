# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai streaming."""

from google.genai import Client, types

client = Client()
for chunk in client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents="Say this is a test",
    config=types.GenerateContentConfig(
        system_instruction="You are a helpful assistant.",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        max_output_tokens=100,
        stop_sequences=["END"],
        seed=42,
        presence_penalty=0.5,
        frequency_penalty=0.5,
    ),
):
    _ = chunk
