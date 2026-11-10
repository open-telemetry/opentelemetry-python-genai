# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler
from llama_index.llms.openai import OpenAI

from opentelemetry.instrumentation.genai.llama_index import (
    LlamaIndexInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.test_util_genai.vcr import (
    scrub_response_headers_overwrite,
)

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "test_openai_api_key")


@pytest.fixture
def openai_llm() -> OpenAI:
    return OpenAI(model="gpt-4o-mini", temperature=0.1)


@pytest.fixture
def openai_llm_without_retries() -> OpenAI:
    # The default three retries would replay an error cassette four times.
    return OpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=0)


@pytest.fixture
def framework_span_handler() -> Iterator[SimpleSpanHandler]:
    dispatcher = get_dispatcher()
    handler = SimpleSpanHandler()
    dispatcher.add_span_handler(handler)
    try:
        yield handler
    finally:
        dispatcher.span_handlers.remove(handler)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "openai-organization": "test_openai_org_id",
                "openai-project": "test_openai_project_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
    }


@pytest.fixture
def instrument_llama_index(
    tracer_provider,
    logger_provider,
    meter_provider,
):
    with instrument(
        LlamaIndexInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as instrumentor:
        yield instrumentor


@pytest.fixture
def instrument_llama_index_with_content(
    tracer_provider,
    logger_provider,
    meter_provider,
):
    with instrument(
        LlamaIndexInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="SPAN_ONLY",
    ) as instrumentor:
        yield instrumentor
