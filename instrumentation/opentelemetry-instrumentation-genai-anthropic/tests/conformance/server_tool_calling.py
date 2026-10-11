# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Anthropic chat with server-side tool calls."""

from __future__ import annotations

from anthropic import Anthropic

Anthropic().messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "Search for OpenTelemetry."}],
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 1,
        }
    ],
)
