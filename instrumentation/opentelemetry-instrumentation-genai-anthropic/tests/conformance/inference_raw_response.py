# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: anthropic chat via with_raw_response."""

from anthropic import Anthropic

client = Anthropic()
raw_response = client.messages.with_raw_response.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Say hello in one word.",
        }
    ],
)
raw_response.parse()
