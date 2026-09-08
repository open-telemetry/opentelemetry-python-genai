# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic streaming chat (inference)."""

from anthropic import Anthropic

client = Anthropic()
with client.messages.create(
    model="claude-sonnet-4-20250514",
    system="You are a helpful assistant.",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one word.",
        }
    ],
    max_tokens=100,
    stop_sequences=["END"],
    extra_body={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    },
    stream=True,
) as stream:
    for _ in stream:
        pass
