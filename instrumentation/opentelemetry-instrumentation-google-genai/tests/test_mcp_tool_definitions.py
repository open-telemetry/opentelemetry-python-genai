# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import wraps
from unittest.mock import AsyncMock

import pytest
from google.genai import Client
from google.genai.models import AsyncModels
from google.genai.types import GenerateContentConfig, GenerateContentResponse

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
    generate_content,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _FakeMcpClientSession:
    def __init__(self) -> None:
        self.list_tools = AsyncMock(side_effect=RuntimeError("boom"))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_mcp_list_tools_failure_does_not_fail_the_call(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    stream: bool,
) -> None:
    monkeypatch.setattr(generate_content, "_is_mcp_imported", True)
    monkeypatch.setattr(
        generate_content, "McpClientSession", _FakeMcpClientSession
    )
    session = _FakeMcpClientSession()
    response = GenerateContentResponse(candidates=[])
    method_name = "generate_content_stream" if stream else "generate_content"

    async def _stream():
        yield response

    @wraps(getattr(AsyncModels, method_name))
    async def call(*args: object, **kwargs: object) -> object:
        return _stream() if stream else response

    monkeypatch.setattr(AsyncModels, method_name, call)
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        models = Client(api_key="test-key").aio.models
        result = await getattr(models, method_name)(
            model="gemini-2.5-flash",
            contents="Hello",
            config=GenerateContentConfig.model_construct(tools=[session]),
        )
        if stream:
            assert [chunk async for chunk in result] == [response]
        else:
            assert result is response

    session.list_tools.assert_awaited_once()
    (span,) = span_exporter.get_finished_spans()
    assert span.status.is_ok
    assert "gen_ai.tool.definitions" not in (span.attributes or {})
