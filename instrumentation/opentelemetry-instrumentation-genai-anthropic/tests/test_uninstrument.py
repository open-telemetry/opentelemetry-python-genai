# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests that uninstrument() stops tracing on cached raw-response accessors.

The SDK caches ``with_raw_response`` and ``with_streaming_response`` per
client together with the wrapped ``create``. Without the ``_active()``
guard on the wrap sites, a client that used an accessor while
instrumented kept tracing through it after ``uninstrument()``
(issue #892); these tests pin the fixed behavior.
"""

import inspect

import anthropic._base_client as base_client
import pytest
from anthropic import Anthropic, AsyncAnthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.test_util_genai.instrumentor import instrument

# Newer anthropic releases vendor httpx as httpx2.
httpx = getattr(base_client, "httpx2", None) or base_client.httpx

_ANSWER = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-x",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_KW = dict(
    model="claude-x",
    max_tokens=10,
    messages=[{"role": "user", "content": "hi"}],
)


def _transport():
    return httpx.MockTransport(
        lambda request: httpx.Response(200, json=_ANSWER)
    )


def _client():
    return Anthropic(
        api_key="x",
        base_url="http://mock",
        max_retries=0,
        http_client=httpx.Client(transport=_transport()),
    )


def _async_client():
    return AsyncAnthropic(
        api_key="x",
        base_url="http://mock",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=_transport()),
    )


def _call_sync(client: Anthropic, accessor: str) -> None:
    """Go through the cached accessor exactly like a caller would."""
    raw = getattr(client.messages, accessor).create(**_KW)
    if accessor == "with_streaming_response":
        # The streaming accessor hands back a ResponseContextManager that
        # yields the raw response on enter.
        with raw as response:
            response.parse()
    else:
        raw.parse()


async def _call_async(client: AsyncAnthropic, accessor: str) -> None:
    """Async counterpart to :func:`_call_sync`."""
    if accessor == "with_streaming_response":
        # AsyncResponseContextManager is entered, not awaited.
        async with getattr(client.messages, accessor).create(
            **_KW
        ) as response:
            await response.parse()
    else:
        raw = await getattr(client.messages, accessor).create(**_KW)
        parsed = raw.parse()
        # Older anthropic versions return the parsed message directly from
        # the async raw-response accessor instead of a coroutine.
        if inspect.isawaitable(parsed):
            await parsed


@pytest.mark.parametrize(
    "accessor", ["with_raw_response", "with_streaming_response"]
)
def test_uninstrument_stops_tracing_cached_raw_response_accessors(
    accessor, span_exporter, tracer_provider, logger_provider, meter_provider
):
    client = _client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        # The SDK caches the accessor, along with the wrapped create.
        _call_sync(client, accessor)
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    _call_sync(client, accessor)
    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "accessor", ["with_raw_response", "with_streaming_response"]
)
async def test_uninstrument_stops_tracing_cached_async_raw_response_accessors(
    accessor, span_exporter, tracer_provider, logger_provider, meter_provider
):
    client = _async_client()
    with instrument(
        AnthropicInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        await _call_async(client, accessor)
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    await _call_async(client, accessor)
    assert len(span_exporter.get_finished_spans()) == 0
