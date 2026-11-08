# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for the Ollama instrumentation.

The cassettes were recorded against a local Ollama server. To re-record one,
delete it, start ``ollama serve`` with the model the test uses pulled, and run
the test with ``--vcr-record=once``.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

import pytest

from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]

HOST = "http://localhost:11434"
CHAT_MODEL = "qwen2.5:0.5b"
REASONING_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [("user-agent", "ollama-python")],
        "decode_compressed_response": True,
    }


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        OllamaInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_with_content(tracer_provider, logger_provider, meter_provider):
    with instrument(
        OllamaInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
