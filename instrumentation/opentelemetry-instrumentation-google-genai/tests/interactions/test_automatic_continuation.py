# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""``interactions.create`` across automatic continuation hops.

google-genai >= 2.34 resumes a generation that hit the per-request decode
limit by sending further requests ("hops") inside one ``create()`` call. These
tests drive the real SDK over a mock transport so its continuation loop runs.
Tests that need automatic continuation skip on older SDKs; the raw-response
tests run on every SDK that has the Interactions API.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

try:
    import httpx
except ImportError:
    import httpx2 as httpx
import pytest
from google.genai import Client
from google.genai.types import HttpOptions

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
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

if not _HAS_INTERACTIONS:
    pytest.skip("This SDK has no Interactions API", allow_module_level=True)

try:
    # Google GenAI >= 2.34.0
    from google.genai.errors import ContinuationInterruptedError

    _HAS_CONTINUATION = True
except ImportError:
    _HAS_CONTINUATION = False

    class ContinuationInterruptedError(Exception):
        pass


try:
    # Google GenAI >= 2.9.0 validates `stream` as a boolean; older SDKs
    # stream on any truthy value.
    import google.genai._gaos.interactions  # noqa: F401

    _SDK_VALIDATES_STREAM = True
except ImportError:
    _SDK_VALIDATES_STREAM = False

requires_continuation = pytest.mark.skipif(
    not _HAS_CONTINUATION, reason="This SDK has no automatic continuation"
)

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
        interaction["steps"] = hop.get(
            "steps",
            [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": hop["text"]}],
                }
            ],
        )
    return interaction


def _text_step(index: int, text: str) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "step.start",
            "index": index,
            "step": {"type": "model_output"},
        },
        {
            "event_type": "step.delta",
            "index": index,
            "delta": {"type": "text", "text": text},
        },
        {"event_type": "step.stop", "index": index},
    ]


def _thought_step(index: int) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "step.start",
            "index": index,
            "step": {"type": "thought"},
        },
        {
            "event_type": "step.delta",
            "index": index,
            "delta": {"type": "thought_signature", "signature": "c2ln"},
        },
        {"event_type": "step.stop", "index": index},
    ]


def _function_call_step(index: int) -> list[dict[str, Any]]:
    return [
        {
            "event_type": "step.start",
            "index": index,
            "step": {
                "type": "function_call",
                "id": "call-1",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
        },
        {"event_type": "step.stop", "index": index},
    ]


def _sse(hop: dict[str, Any]) -> bytes:
    if "checkpoint" in hop:
        # The hop streams up to a checkpoint, then the connection drops before
        # its completion event.
        ending = [
            {
                "event_type": "interaction.status_update",
                "interaction_id": hop["id"],
                "status": "in_progress",
                "continuation_token": hop["checkpoint"],
            }
        ]
    else:
        ending = [
            {
                "event_type": "interaction.completed",
                "interaction": _interaction(
                    hop, with_steps=hop.get("completion_steps", False)
                ),
            }
        ]
    events = [
        {
            "event_type": "interaction.created",
            "interaction": {
                "id": hop["id"],
                "object": "interaction",
                "model": "gemini-2.5-flash",
            },
        },
        *hop.get("step_events", _text_step(0, hop["text"])),
        *ending,
    ]
    return "".join(
        f"event: {event['event_type']}\ndata: {json.dumps(event)}\n\n"
        for event in events
    ).encode()


class _DroppedStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """A response body whose connection drops after ``data`` is sent."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def __iter__(self) -> Iterator[bytes]:
        yield self._data
        raise httpx.ReadError("connection dropped")

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data
        raise httpx.ReadError("connection dropped")


class _Server:
    """Answers each hop in turn; a hop past ``fail_from`` gets a 400."""

    def __init__(
        self,
        fail_from: int | None = None,
        hops: list[dict[str, Any]] | None = None,
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._fail_from = fail_from
        self._hops = hops if hops is not None else _HOPS

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        index = len(self.requests)
        self.requests.append(body)
        if self._fail_from is not None and index >= self._fail_from:
            return httpx.Response(
                400,
                json={"error": {"code": 400, "message": "bad hop"}},
            )
        hop = self._hops[index]
        if body.get("stream"):
            headers = {"content-type": "text/event-stream"}
            if "checkpoint" in hop:
                return httpx.Response(
                    200, headers=headers, stream=_DroppedStream(_sse(hop))
                )
            return httpx.Response(200, headers=headers, content=_sse(hop))
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
        GenAIAttributes.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS: 4,
        "gen_ai.usage.text.input_tokens": 25,
        "gen_ai.usage.text.output_tokens": 12,
    }
    for name, value in expected.items():
        assert attributes[name] == value, name
        assert type(attributes[name]) is int, name


@requires_continuation
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
    assert span.name == "chat gemini-2.5-flash"
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
        output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        text = "".join(
            part["content"]
            for message in output
            for part in message["parts"]
            if part["type"] == "text"
        )
        assert text == "Hello world"


@requires_continuation
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_interrupted_continuation_keeps_completed_hops(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
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
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
        return
    output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert [message["role"] for message in output] == ["assistant"]
    assert output[0]["parts"] == ({"type": "text", "content": "Hello"},)


@requires_continuation
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_interrupted_continuation_records_tool_call_only_output(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
) -> None:
    tool_call = {
        "type": "function_call",
        "id": "call-1",
        "name": "get_weather",
        "arguments": {"city": "Paris"},
    }
    server = _Server(fail_from=1, hops=[{**_HOPS[0], "steps": [tool_call]}])
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        with pytest.raises(ContinuationInterruptedError):
            await _create(
                server,
                asynchronous,
                model="gemini-2.5-flash",
                input="What is the weather in Paris?",
            )

    attributes = _only_span(span_exporter).attributes
    assert attributes is not None
    output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert [message["role"] for message in output] == ["assistant"]
    assert output[0]["parts"] == (
        {
            "type": "tool_call",
            "id": "call-1",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        },
    )


@requires_continuation
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


# Each continuation hop numbers its steps from 0 again, so the second hop
# reuses indices the first hop already used.
_HOP_LAYOUTS = {
    "text_then_thought": (
        _text_step(0, "Hello"),
        [*_thought_step(0), *_text_step(1, " world")],
        [
            {"type": "text", "content": "Hello"},
            {"type": "text", "content": " world"},
        ],
    ),
    "tool_call_then_text": (
        [*_text_step(0, "Checking."), *_function_call_step(1)],
        _text_step(0, "Sunny."),
        [
            {"type": "text", "content": "Checking."},
            {
                "type": "tool_call",
                "id": "call-1",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
            {"type": "text", "content": "Sunny."},
        ],
    ),
}


@requires_continuation
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_keeps_output_across_checkpoint_resume(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
    asynchronous: bool,
) -> None:
    # The first hop drops after a checkpoint, so it never completes; the SDK
    # resumes from the checkpoint in a new hop that reuses step index 0.
    server = _Server(
        hops=[
            {
                **_HOPS[0],
                "checkpoint": "checkpoint-1",
                "step_events": _text_step(0, "Hello"),
            },
            {
                **_HOPS[1],
                "step_events": [*_thought_step(0), *_text_step(1, " world")],
            },
        ]
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture=content_mode,
    ):
        await _create(
            server,
            asynchronous,
            model="gemini-2.5-flash",
            input="Say hello",
            stream=True,
        )

    assert len(server.requests) == 2
    assert server.requests[1]["continuation_token"] == "checkpoint-1"
    span = _only_span(span_exporter)
    assert span.status.status_code == StatusCode.UNSET
    attributes = span.attributes
    assert attributes is not None
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
        return
    output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert [message["role"] for message in output] == ["assistant"]
    assert output[0]["parts"] == (
        {"type": "text", "content": "Hello"},
        {"type": "text", "content": " world"},
    )


@requires_continuation
@pytest.mark.parametrize("step_deltas", [True, False])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_keeps_each_hop_output_when_completion_carries_steps(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
    asynchronous: bool,
    step_deltas: bool,
) -> None:
    # Each hop's completion event carries that hop's steps, with or without
    # the step events streamed before it.
    hops = [{**hop, "completion_steps": True} for hop in _HOPS]
    if not step_deltas:
        hops = [{**hop, "step_events": []} for hop in hops]
    server = _Server(hops=hops)
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture=content_mode,
    ):
        await _create(
            server,
            asynchronous,
            model="gemini-2.5-flash",
            input="Say hello",
            stream=True,
        )

    assert len(server.requests) == 2
    attributes = _only_span(span_exporter).attributes
    assert attributes is not None
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
        return
    output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert [message["role"] for message in output] == ["assistant"]
    assert output[0]["parts"] == (
        {"type": "text", "content": "Hello"},
        {"type": "text", "content": " world"},
    )


@requires_continuation
@pytest.mark.parametrize("layout", sorted(_HOP_LAYOUTS))
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_keeps_each_hop_output_when_step_indices_repeat(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    content_mode: str,
    asynchronous: bool,
    layout: str,
) -> None:
    first, second, expected_parts = _HOP_LAYOUTS[layout]
    server = _Server(
        hops=[
            {**_HOPS[0], "step_events": first},
            {**_HOPS[1], "step_events": second},
        ]
    )
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture=content_mode,
    ):
        await _create(
            server,
            asynchronous,
            model="gemini-2.5-flash",
            input="What is the weather in Paris?",
            stream=True,
        )

    assert len(server.requests) == 2
    attributes = _only_span(span_exporter).attributes
    assert attributes is not None
    if content_mode == "NO_CONTENT":
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in attributes
        return
    output = attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    assert [message["role"] for message in output] == ["assistant"]
    assert output[0]["parts"] == tuple(expected_parts)


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
    assert span.name == "chat gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS not in attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in attributes
    assert GenAIAttributes.GEN_AI_REQUEST_STREAM not in attributes


@pytest.mark.parametrize("stream_value", [True, 1, "true", False, 0, "false"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("accessor", ["raw", "streaming"])
@pytest.mark.asyncio
async def test_raw_response_records_request_stream(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
    accessor: str,
    stream_value: object,
) -> None:
    streaming = (
        stream_value in (True, 1, "true")
        if _SDK_VALIDATES_STREAM
        else bool(stream_value)
    )
    server = _Server()
    params = {
        "model": "gemini-2.5-flash",
        "input": "Say hello",
        "stream": stream_value,
    }
    if asynchronous:
        async with _async_client(server) as client:
            interactions = client.aio.interactions
            if accessor == "raw":
                response = await interactions.with_raw_response.create(
                    **params
                )
                await response.close()
            else:
                async with interactions.with_streaming_response.create(
                    **params
                ):
                    pass
    else:
        with _sync_client(server) as client:
            interactions = client.interactions
            if accessor == "raw":
                interactions.with_raw_response.create(**params).close()
            else:
                with interactions.with_streaming_response.create(**params):
                    pass

    assert len(server.requests) == 1
    attributes = _only_span(span_exporter).attributes
    assert attributes is not None
    if streaming:
        assert attributes[GenAIAttributes.GEN_AI_REQUEST_STREAM] is True
    else:
        assert GenAIAttributes.GEN_AI_REQUEST_STREAM not in attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in attributes


@requires_continuation
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_stream_closed_early_ends_span(
    instrumented: None,
    span_exporter: InMemorySpanExporter,
    asynchronous: bool,
) -> None:
    # The SDK replaces the stream's `close` to stop continuation; closing must
    # not recurse through the instrumentation wrapper (#1032).
    server = _Server()
    params = {
        "model": "gemini-2.5-flash",
        "input": "Say hello",
        "stream": True,
    }
    if asynchronous:
        async with _async_client(server) as client:
            stream = await client.aio.interactions.create(**params)
            async for _ in stream:
                break
            await stream.close()
    else:
        with _sync_client(server) as client:
            stream = client.interactions.create(**params)
            for _ in stream:
                break
            stream.close()

    assert len(server.requests) == 1
    span = _only_span(span_exporter)
    attributes = span.attributes
    assert attributes is not None
    assert span.name == "chat gemini-2.5-flash"
    assert span.status.status_code == StatusCode.UNSET
    assert ERROR_TYPE not in attributes


@requires_continuation
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
    assert attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES] == (
        {
            "role": "user",
            "parts": (
                {"type": "text", "content": "Say"},
                {"type": "text", "content": "hello"},
            ),
        },
    )
