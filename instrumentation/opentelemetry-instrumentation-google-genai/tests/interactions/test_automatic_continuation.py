# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``interactions.create`` across automatic continuation hops.

google-genai >= 2.34 resumes a generation that hit the per-request decode
limit by sending further requests ("hops") inside one ``create()`` call. These
tests drive the real SDK over a mock transport so its continuation loop runs.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx
import pytest
from google.genai import Client
from google.genai.types import HttpOptions

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import StatusCode

try:
    from google.genai.errors import ContinuationInterruptedError
except ImportError:
    pytest.skip(
        "This SDK has no automatic continuation", allow_module_level=True
    )

_INITIAL_INPUT_TOKENS = "gcp.gen_ai.usage.initial_input_tokens"

_HOPS: list[dict[str, Any]] = [
    {
        "id": "v1_hop_1",
        "status": "requires_continuation",
        "continuation_token": "token-1",
        "text": "Hello",
        "usage": {
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_thought_tokens": 3,
            "total_cached_tokens": 4,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 10}],
            "output_tokens_by_modality": [{"modality": "text", "tokens": 5}],
        },
    },
    {
        "id": "v1_hop_2",
        "status": "completed",
        "text": " world",
        "usage": {
            "total_input_tokens": 15,
            "total_output_tokens": 7,
            "total_thought_tokens": 1,
            "total_cached_tokens": 6,
            "input_tokens_by_modality": [{"modality": "text", "tokens": 15}],
            "output_tokens_by_modality": [{"modality": "text", "tokens": 7}],
        },
    },
]


def _interaction(hop: dict[str, Any], *, with_steps: bool) -> dict[str, Any]:
    interaction: dict[str, Any] = {
        "id": hop["id"],
        "object": "interaction",
        "model": "gemini-2.5-flash",
        "status": hop["status"],
        "usage": hop["usage"],
    }
    if "continuation_token" in hop:
        interaction["continuation_token"] = hop["continuation_token"]
    if with_steps:
        interaction["steps"] = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": hop["text"]}],
            }
        ]
    return interaction


def _sse(hop: dict[str, Any]) -> bytes:
    events = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": hop["id"],
                "object": "interaction",
                "model": "gemini-2.5-flash",
            },
        },
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "model_output"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "text", "text": hop["text"]},
        },
        {"event_type": "step.stop", "index": 0},
        {
            "event_type": "interaction.completed",
            "interaction": _interaction(hop, with_steps=False),
        },
    ]
    return "".join(
        f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    ).encode()


class _Server:
    """Answers each hop in turn; a hop past ``fail_from`` gets a 400."""

    def __init__(self, fail_from: int | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._fail_from = fail_from

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        index = len(self.requests)
        self.requests.append(body)
        if self._fail_from is not None and index >= self._fail_from:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "bad hop"}},
            )
        hop = _HOPS[index]
        if body.get("stream"):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(hop),
            )
        return httpx.Response(200, json=_interaction(hop, with_steps=True))


@contextmanager
def _sync_client(server: _Server) -> Iterator[Client]:
    with httpx.Client(transport=httpx.MockTransport(server)) as transport:
        with Client(
            api_key="test-key",
            vertexai=False,
            http_options=HttpOptions(httpx_client=transport),
        ) as client:
            yield client


@asynccontextmanager
async def _async_client(server: _Server) -> AsyncIterator[Client]:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server)
    ) as transport:
        with Client(
            api_key="test-key",
            vertexai=False,
            http_options=HttpOptions(httpx_async_client=transport),
        ) as client:
            yield client


async def _create(
    server: _Server, asynchronous: bool, **params: Any
) -> list[Any] | Any:
    """Run one ``create()``, draining it when streamed."""
    if asynchronous:
        async with _async_client(server) as client:
            result = await client.aio.interactions.create(**params)
            if params.get("stream"):
                return [event async for event in result]
            return result
    with _sync_client(server) as client:
        result = client.interactions.create(**params)
        if params.get("stream"):
            return list(result)
        return result


@pytest.fixture(name="content_mode", params=["NO_CONTENT", "SPAN_ONLY"])
def fixture_content_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(name="instrumented")
def fixture_instrumented(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    content_mode: str,
) -> Iterator[None]:
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture=content_mode,
    ):
        yield


def _only_span(span_exporter: InMemorySpanExporter) -> ReadableSpan:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    return span


def _assert_summed_usage(span: ReadableSpan) -> None:
    attributes = span.attributes
    assert attributes is not None
    expected = {
        GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 25,
        GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 12,
        GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS: 10,
        _INITIAL_INPUT_TOKENS: 10,
    }
    for name, value in expected.items():
        assert attributes[name] == value, name
        assert type(attributes[name]) is int, name


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_two_hops_produce_one_span_with_summed_usage(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
    asynchronous: bool,
    stream: bool,
) -> None:
    server = _Server()
    result = await _create(
        server,
        asynchronous,
        model="gemini-2.5-flash",
        input="Say hello",
        stream=stream,
    )

    assert len(server.requests) == 2
    assert server.requests[1]["continuation_token"] == "token-1"
    if stream:
        completed = [
            event
            for event in result
            if event.event_type == "interaction.completed"
        ]
        assert len(completed) == 2
    else:
        assert result.id == "v1_hop_2"

    span = _only_span(span_exporter)
    assert span.name == "interactions.create gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET
    attributes = span.attributes
    assert attributes is not None
    _assert_summed_usage(span)
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "v1_hop_2"
    assert attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
    else:
        output = json.loads(
            str(attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES])
        )
        text = "".join(
            part["content"]
            for message in output
            for part in message["parts"]
            if part["type"] == "text"
        )
        assert text == "Hello world"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_interrupted_continuation_keeps_completed_hop_usage(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    stream: bool,
) -> None:
    server = _Server(fail_from=1)
    with pytest.raises(ContinuationInterruptedError):
        await _create(
            server,
            asynchronous,
            model="gemini-2.5-flash",
            input="Say hello",
            stream=stream,
        )

    assert len(server.requests) == 2
    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    attributes = span.attributes
    assert attributes is not None
    assert (
        attributes[ERROR_TYPE]
        == "google.genai.errors.ContinuationInterruptedError"
    )
    assert attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert attributes[_INITIAL_INPUT_TOKENS] == 10


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_disabled_continuation_records_one_span_per_call(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    stream: bool,
) -> None:
    server = _Server()
    params: dict[str, Any] = {
        "model": "gemini-2.5-flash",
        "stream": stream,
        "automatic_continuation": False,
    }
    with warnings.catch_warnings():
        # The SDK warns, once per process, that the generation was left
        # incomplete.
        warnings.simplefilter("ignore", UserWarning)
        await _create(server, asynchronous, input="Say hello", **params)
    await _create(
        server,
        asynchronous,
        previous_interaction_id="v1_hop_1",
        continuation_token="token-1",
        **params,
    )

    first, second = span_exporter.get_finished_spans()
    for span, hop in zip((first, second), _HOPS):
        attributes = span.attributes
        assert attributes is not None
        assert attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == hop["id"]
        assert (
            attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
            == hop["usage"]["total_input_tokens"]
        )
        assert (
            attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
            == hop["usage"]["total_output_tokens"]
        )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("accessor", ["raw", "streaming"])
@pytest.mark.asyncio
async def test_raw_response_records_request_only_span(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    accessor: str,
) -> None:
    server = _Server()
    params = {"model": "gemini-2.5-flash", "input": "Say hello"}
    if asynchronous:
        async with _async_client(server) as client:
            interactions = client.aio.interactions
            if accessor == "raw":
                response = await interactions.with_raw_response.create(
                    **params
                )
                parsed = await response.parse()
            else:
                async with interactions.with_streaming_response.create(
                    **params
                ) as response:
                    parsed = await response.parse()
    else:
        with _sync_client(server) as client:
            interactions = client.interactions
            if accessor == "raw":
                response = interactions.with_raw_response.create(**params)
                parsed = response.parse()
            else:
                with interactions.with_streaming_response.create(
                    **params
                ) as response:
                    parsed = response.parse()

    # The raw accessors bypass automatic continuation.
    assert len(server.requests) == 1
    assert parsed.id == "v1_hop_1"
    span = _only_span(span_exporter)
    attributes = span.attributes
    assert attributes is not None
    assert span.name == "interactions.create gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in attributes


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_untyped_content_blocks_are_recorded_as_text(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
    asynchronous: bool,
) -> None:
    server = _Server(fail_from=2)
    await _create(
        server,
        asynchronous,
        model="gemini-2.5-flash",
        input=[{"text": "Say"}, {"type": "text", "text": "hello"}],
    )

    assert server.requests[0]["input"] == [
        {
            "type": "user_input",
            "content": [
                {"type": "text", "text": "Say"},
                {"type": "text", "text": "hello"},
            ],
        }
    ]
    attributes = _only_span(span_exporter).attributes
    assert attributes is not None
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in attributes
        return
    assert json.loads(
        str(attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES])
    ) == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "Say"},
                {"type": "text", "content": "hello"},
            ],
            "name": None,
        }
    ]
