# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Google GenAI interaction retrieval."""

from __future__ import annotations

from google.genai import Client

client = Client(api_key="test_google_genai_api_key", vertexai=False)
client.interactions.get("interaction-fetch-1")
