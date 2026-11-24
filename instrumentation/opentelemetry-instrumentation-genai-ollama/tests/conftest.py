# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for the Ollama instrumentation.

The cassettes were recorded against a local Ollama server. To re-record one,
delete it, start ``ollama serve`` with the model the test uses pulled, and run
the test with ``--vcr-record=once``.
"""
# pylint: disable=redefined-outer-name

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from opentelemetry import trace
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


@pytest.fixture
def request_span_capture() -> tuple[
    list[int],
    Callable[[Any], None],
    Callable[[Any], Awaitable[None]],
]:
    """Span ids current when each HTTP request is sent, plus sync and async
    ``event_hooks`` that record them."""
    seen: list[int] = []

    def hook(_request: Any) -> None:
        seen.append(trace.get_current_span().get_span_context().span_id)

    async def async_hook(_request: Any) -> None:
        hook(_request)

    return seen, hook, async_hook
