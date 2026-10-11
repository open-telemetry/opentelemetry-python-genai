# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API multi-turn conversation."""

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

client = OpenAI()
first = client.responses.create(
    model=DEFAULT_MODEL,
    input="Remember that my favorite color is blue.",
)

client.responses.create(
    model=DEFAULT_MODEL,
    input="What is my favorite color?",
    previous_response_id=first.id,
)
