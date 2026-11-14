# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import anthropic
import anthropic._base_client as base_client
import pytest

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument

httpx = getattr(base_client, "httpx2", None) or base_client.httpx

_ANSWER = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_KW: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 10,
    "messages": [{"role": "user", "content": "hi"}],
}


def _transport() -> Any:
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json=_ANSWER)
    )


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="x",
        http_client=httpx.Client(transport=_transport()),
    )


def _async_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        api_key="x",
        http_client=httpx.AsyncClient(transport=_transport()),
    )


def test_uninstrument_stops_tracing_cached_with_raw_response(
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
) -> None:
    client = _client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        client.messages.with_raw_response.create(**_KW)
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    client.messages.with_raw_response.create(**_KW)
    assert len(span_exporter.get_finished_spans()) == 0


def test_uninstrument_stops_tracing_cached_with_streaming_response(
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
) -> None:
    client = _client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        with client.messages.with_streaming_response.create(**_KW) as r:
            r.parse()
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    with client.messages.with_streaming_response.create(**_KW) as r:
        r.parse()
    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_uninstrument_stops_tracing_cached_async_with_raw_response(
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
) -> None:
    client = _async_client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        await client.messages.with_raw_response.create(**_KW)
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    await client.messages.with_raw_response.create(**_KW)
    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
async def test_uninstrument_stops_tracing_cached_async_with_streaming_response(
    span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    logger_provider: Any,
    meter_provider: Any,
) -> None:
    client = _async_client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        async with client.messages.with_streaming_response.create(**_KW) as r:
            await r.parse()
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    async with client.messages.with_streaming_response.create(**_KW) as r:
        await r.parse()
    assert len(span_exporter.get_finished_spans()) == 0
