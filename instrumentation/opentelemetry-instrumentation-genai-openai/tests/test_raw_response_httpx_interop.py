# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming ``with_raw_response`` alongside ``opentelemetry-instrumentation-httpx``.

The streaming proxy patches ``close``/``aclose`` on the underlying httpx
response to finalize the span when the caller never parses. httpx
instrumentation patches the *transport*'s ``handle_request``, so the two should
not collide — these tests hold that: both spans are emitted, correctly nested,
on the parse-and-drain path and on the never-parse path where our close
fallback is what ends the span.

httpx instrumentation is enabled per-client here rather than globally so it
stays scoped to these tests.
"""

from __future__ import annotations

import json

import pytest
from openai import AsyncOpenAI, OpenAI

# This module tests interop with opentelemetry-instrumentation-httpx, which
# only applies when openai uses httpx (v1/v2). Skip the entire module when
# httpx is not installed (e.g. openai v3+, which uses httpx2 instead).
httpx = pytest.importorskip("httpx")

# Only pinned in requirements.latest.txt; oldest envs skip this module.
HTTPXClientInstrumentor = pytest.importorskip(
    "opentelemetry.instrumentation.httpx"
).HTTPXClientInstrumentor

_HTTPX_SPAN = "POST"
_GENAI_SPAN = "chat gpt-4"


def _chunk(delta: dict | None = None, finish: str | None = None) -> str:
    return json.dumps(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "finish_reason": finish,
                }
            ],
        }
    )


_SSE_BODY = (
    f"data: {_chunk({'content': 'hi'})}\n\n"
    f"data: {_chunk(finish='stop')}\n\n"
    "data: [DONE]\n\n"
).encode()


def _sse_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_SSE_BODY,
    )


@pytest.fixture(autouse=True)
def fixture_vcr():
    """Requests are served by a mock transport, so no cassettes here."""
    yield


@pytest.fixture
def httpx_instrumented_client(tracer_provider):
    client = httpx.Client(transport=httpx.MockTransport(_sse_response))
    HTTPXClientInstrumentor.instrument_client(
        client, tracer_provider=tracer_provider
    )
    yield client
    HTTPXClientInstrumentor.uninstrument_client(client)


@pytest.fixture
def async_httpx_instrumented_client(tracer_provider):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_sse_response),
    )
    HTTPXClientInstrumentor.instrument_client(
        client, tracer_provider=tracer_provider
    )
    yield client
    HTTPXClientInstrumentor.uninstrument_client(client)


def _create_stream(client):
    return client.chat.completions.with_raw_response.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )


def _assert_both_spans_nested(span_exporter):
    spans = {span.name: span for span in span_exporter.get_finished_spans()}
    assert set(spans) == {_HTTPX_SPAN, _GENAI_SPAN}

    genai_span, httpx_span = spans[_GENAI_SPAN], spans[_HTTPX_SPAN]
    assert httpx_span.context.trace_id == genai_span.context.trace_id
    assert httpx_span.parent is not None
    assert httpx_span.parent.span_id == genai_span.context.span_id


def test_raw_response_stream_parsed_emits_both_spans(
    span_exporter, instrument_with_content, httpx_instrumented_client
):
    client = OpenAI(
        api_key="test",
        base_url="http://test",
        http_client=httpx_instrumented_client,
    )

    for _ in _create_stream(client).parse():
        pass

    _assert_both_spans_nested(span_exporter)


def test_raw_response_stream_closed_without_parse_emits_both_spans(
    span_exporter, instrument_with_content, httpx_instrumented_client
):
    # The close fallback owns finalization here — the path where a conflict
    # with httpx's patching would surface.
    client = OpenAI(
        api_key="test",
        base_url="http://test",
        http_client=httpx_instrumented_client,
    )

    raw = _create_stream(client)
    assert [s.name for s in span_exporter.get_finished_spans()] == [
        _HTTPX_SPAN
    ]  # genai span still open

    raw.http_response.close()

    _assert_both_spans_nested(span_exporter)


@pytest.mark.asyncio
async def test_async_raw_response_stream_parsed_emits_both_spans(
    span_exporter, instrument_with_content, async_httpx_instrumented_client
):
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test",
        http_client=async_httpx_instrumented_client,
    )

    async for _ in (await _create_stream(client)).parse():
        pass

    _assert_both_spans_nested(span_exporter)


@pytest.mark.asyncio
async def test_async_raw_response_stream_closed_without_parse_emits_both_spans(
    span_exporter, instrument_with_content, async_httpx_instrumented_client
):
    client = AsyncOpenAI(
        api_key="test",
        base_url="http://test",
        http_client=async_httpx_instrumented_client,
    )

    raw = await _create_stream(client)
    await raw.http_response.aclose()

    _assert_both_spans_nested(span_exporter)
