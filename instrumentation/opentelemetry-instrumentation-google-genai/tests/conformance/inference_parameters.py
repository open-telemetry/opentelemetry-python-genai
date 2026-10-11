# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: google-genai chat completion (inference)."""

from __future__ import annotations

import sys
from pathlib import Path

from google.genai import Client

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interactions.util import create_request_parameters

client = Client(api_key="test_google_genai_api_key", vertexai=False)
parameters, _ = create_request_parameters()
client.interactions.create(
    model="gemini-2.5-flash",
    input="Hello, how can you help me today?",
    **parameters,
)
