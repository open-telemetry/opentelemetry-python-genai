# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai chat completion (inference)."""

from openai import OpenAI

OpenAI().chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say this is a test"},
    ],
    model="gpt-4o-mini",
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
    stop=["END"],
    seed=42,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    stream=False,
)
