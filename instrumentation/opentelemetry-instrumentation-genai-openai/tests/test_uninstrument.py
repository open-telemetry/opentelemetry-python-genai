# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect

import openai
import openai._base_client as base_client

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

# Newer openai releases vendor httpx as httpx2.
httpx = getattr(base_client, "httpx2", None) or base_client.httpx

_ANSWER = {
    "id": "c",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        }
    ],
}
_MESSAGES = [{"role": "user", "content": "hi"}]


def _transport():
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json=_ANSWER)
    )


def _client():
    return openai.OpenAI(
        api_key="x",
        base_url="http://mock/v1",
        max_retries=0,
        http_client=httpx.Client(transport=_transport()),
    )


def _async_client():
    return openai.AsyncOpenAI(
        api_key="x",
        base_url="http://mock/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_transport()),
    )


def test_uninstrument_stops_tracing_cached_with_raw_response(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    client = _client()
    with instrument(
        OpenAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        # The SDK caches with_raw_response, along with the wrapped create.
        client.chat.completions.with_raw_response.create(
            model="m", messages=_MESSAGES
        ).parse()
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    client.chat.completions.with_raw_response.create(
        model="m", messages=_MESSAGES
    ).parse()
    assert len(span_exporter.get_finished_spans()) == 0


def test_uninstrument_stops_tracing_cached_async_with_raw_response(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    async def create(client):
        raw = await client.chat.completions.with_raw_response.create(
            model="m", messages=_MESSAGES
        )
        # LegacyAPIResponse.parse() is synchronous even for the async client.
        parsed = raw.parse()
        return await parsed if inspect.isawaitable(parsed) else parsed

    client = _async_client()
    with instrument(
        OpenAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        asyncio.run(create(client))
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    asyncio.run(create(client))
    assert len(span_exporter.get_finished_spans()) == 0
