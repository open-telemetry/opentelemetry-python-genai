# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API streaming."""

from openai import OpenAI

DEFAULT_MODEL = "gpt-4o-mini"

with OpenAI().responses.create(
    model=DEFAULT_MODEL,
    instructions="You are a helpful assistant.",
    input="Say this is a test",
    stream=True,
) as stream:
    for _ in stream:
        pass
