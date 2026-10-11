# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai embeddings."""

from google.genai import Client

client = Client()
client.models.embed_content(
    model="gemini-embedding-2",
    contents="The quick brown fox jumps over the lazy dog",
)
