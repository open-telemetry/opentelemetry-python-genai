# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API hosted tool calling."""

from __future__ import annotations

from openai import OpenAI

OpenAI().responses.create(
    model="gpt-4o-mini",
    input="Search for OpenTelemetry.",
    tools=[{"type": "web_search"}],
)
