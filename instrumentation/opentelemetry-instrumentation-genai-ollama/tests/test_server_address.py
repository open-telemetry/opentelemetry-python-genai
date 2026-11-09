# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the server address and port taken from the client's host."""

from __future__ import annotations

import ollama
import pytest

from opentelemetry.instrumentation.genai.ollama.utils import (
    server_address_and_port,
)


@pytest.mark.parametrize(
    "host, expected",
    [
        ("http://localhost:11434", ("localhost", 11434)),
        ("localhost", ("localhost", 11434)),
        ("http://my-server:8080", ("my-server", 8080)),
        # Without an explicit port, the scheme's default is used, not 11434.
        ("https://ollama.com", ("ollama.com", None)),
        ("http://my-server", ("my-server", None)),
    ],
)
def test_server_address_and_port(host, expected):
    assert server_address_and_port(ollama.Client(host=host)) == expected
