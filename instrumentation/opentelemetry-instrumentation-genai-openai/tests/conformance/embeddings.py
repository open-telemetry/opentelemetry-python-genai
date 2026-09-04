# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai embedding."""

from openai import OpenAI

OpenAI().embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox jumps over the lazy dog",
    encoding_format="float",
    dimensions=256,
)
