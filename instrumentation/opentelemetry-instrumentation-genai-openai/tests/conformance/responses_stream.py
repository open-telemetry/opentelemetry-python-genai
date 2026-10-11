# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API stream helper."""

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"
SYSTEM_INSTRUCTIONS = "You are a helpful assistant."
USER_PROMPT = "Say this is a test"

with OpenAI().responses.stream(
    model=DEFAULT_MODEL,
    instructions=SYSTEM_INSTRUCTIONS,
    input=USER_PROMPT,
) as stream:
    stream.until_done()
