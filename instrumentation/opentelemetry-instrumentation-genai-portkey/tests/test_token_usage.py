# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from types import SimpleNamespace

import httpx
import portkey_ai
import pytest

from opentelemetry.instrumentation.genai.portkey.utils import (
    set_usage_properties,
)
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_TOKEN_TYPE,
)
from opentelemetry.semconv._incubating.metrics.gen_ai_metrics import (
    GEN_AI_CLIENT_TOKEN_USAGE,
)
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler

_TOTALS = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
}
_AGGREGATES = {
    "gen_ai.usage.input_tokens": 100,
    "gen_ai.usage.output_tokens": 20,
}
_CACHE_USAGE = {
    **_TOTALS,
    "cache_creation_input_tokens": 30,
    "cache_read_input_tokens": 40,
    "prompt_tokens_details": {"cached_tokens": 40},
}
_CACHE_ATTRIBUTES = {
    **_AGGREGATES,
    "gen_ai.usage.cache_write.input_tokens": 30,
    "gen_ai.usage.cache_read.input_tokens": 40,
}
_AUDIO_USAGE = {
    **_TOTALS,
    "prompt_tokens_details": {"audio_tokens": 25, "cached_tokens": 40},
    "completion_tokens_details": {"audio_tokens": 5},
}
_AUDIO_ATTRIBUTES = {
    **_AGGREGATES,
    "gen_ai.usage.cache_read.input_tokens": 40,
    "gen_ai.usage.audio.input_tokens": 25,
    "gen_ai.usage.audio.output_tokens": 5,
}
_IMAGE_OUTPUT_USAGE = {
    **_TOTALS,
    "completion_tokens_details": {"text_tokens": 8, "image_tokens": 12},
}
_IMAGE_OUTPUT_ATTRIBUTES = {
    **_AGGREGATES,
    "gen_ai.usage.text.output_tokens": 8,
    "gen_ai.usage.image.output_tokens": 12,
}
_OPENAI_USAGE = {
    **_TOTALS,
    "prompt_tokens_details": {
        "cache_write_tokens": 30,
        "cached_tokens": 40,
        "text_tokens": 50,
        "image_tokens": 25,
        "audio_tokens": 25,
    },
    "completion_tokens_details": {
        "text_tokens": 15,
        "audio_tokens": 5,
        "reasoning_tokens": 7,
    },
}
_OPENAI_ATTRIBUTES = {
    **_AUDIO_ATTRIBUTES,
    "gen_ai.usage.cache_write.input_tokens": 30,
    "gen_ai.usage.text.input_tokens": 50,
    "gen_ai.usage.image.input_tokens": 25,
    "gen_ai.usage.text.output_tokens": 15,
    "gen_ai.usage.reasoning.output_tokens": 7,
}


@pytest.fixture(params=["chat", "prompt"])
def completion_api(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(
    params=[
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not hasattr(portkey_ai, "AsyncPortkey"),
                reason="AsyncPortkey is unavailable in this SDK version",
            ),
        ),
    ]
)
def asynchronous(request: pytest.FixtureRequest) -> bool:
    return request.param


class _ResponseStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], error: Exception | None) -> None:
        self._chunks = chunks
        self._error = error

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks
        if self._error is not None:
            raise self._error
        yield b"data: [DONE]\n\n"

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self:
            yield chunk


def _mock_http(
    monkeypatch: pytest.MonkeyPatch,
    usages: list[dict[str, object] | None],
    error: Exception | None = None,
) -> None:
    def response(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "portkey.test"
        body = {
            "id": "completion-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "reply"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usages[0],
        }
        if not json.loads(request.content).get("stream"):
            return httpx.Response(200, json=body, request=request)
        chunks = [
            (
                "data: "
                + json.dumps(
                    {
                        **body,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "reply"},
                                "finish_reason": None,
                            }
                        ],
                        "usage": usage,
                    }
                )
                + "\n\n"
            ).encode()
            for usage in usages
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_ResponseStream(chunks, error),
            request=request,
        )

    def handle_request(
        _transport: httpx.HTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        return response(request)

    async def handle_async_request(
        _transport: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        return response(request)

    # Older Portkey versions do not accept an injected HTTP client.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request
    )


def _usage_attributes(
    span_exporter: InMemorySpanExporter,
) -> dict[str, object]:
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    attributes = {
        key: value
        for key, value in span.attributes.items()
        if key.startswith("gen_ai.usage.")
    }
    assert all(type(value) is int for value in attributes.values())
    return attributes


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(_CACHE_USAGE, _CACHE_ATTRIBUTES, id="cache"),
        pytest.param(
            {
                **_TOTALS,
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 40,
            },
            _CACHE_ATTRIBUTES,
            id="cache-without-nested-details",
        ),
        pytest.param(_AUDIO_USAGE, _AUDIO_ATTRIBUTES, id="audio"),
        pytest.param(
            _IMAGE_OUTPUT_USAGE, _IMAGE_OUTPUT_ATTRIBUTES, id="image-output"
        ),
        pytest.param(_OPENAI_USAGE, _OPENAI_ATTRIBUTES, id="openai-details"),
        pytest.param(
            {**_TOTALS, "completion_tokens_details": {"reasoning_tokens": 7}},
            {**_AGGREGATES, "gen_ai.usage.reasoning.output_tokens": 7},
            id="reasoning-only",
        ),
        pytest.param(_TOTALS, _AGGREGATES, id="aggregate-only"),
        pytest.param(None, {}, id="no-usage"),
    ],
)
@pytest.mark.asyncio
async def test_sdk_detailed_usage(
    monkeypatch: pytest.MonkeyPatch,
    instrument_portkey: object,
    span_exporter: InMemorySpanExporter,
    metric_reader: InMemoryMetricReader,
    completion_api: str,
    asynchronous: bool,
    streaming: bool,
    usage: dict[str, object] | None,
    expected: Mapping[str, int],
) -> None:
    _mock_http(monkeypatch, [usage, usage, None])
    client_type = (
        getattr(portkey_ai, "AsyncPortkey")
        if asynchronous
        else portkey_ai.Portkey
    )
    client = client_type(
        api_key="test-key",
        provider="openai",
        base_url="https://portkey.test/v1",
    )
    if completion_api == "chat":
        result = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=streaming,
        )
    else:
        result = client.prompts.completions.create(
            prompt_id="test-prompt", variables={}, stream=streaming
        )
    if asynchronous:
        result = await result

    if streaming:
        assert not span_exporter.get_finished_spans()
        chunks = (
            [chunk async for chunk in result] if asynchronous else list(result)
        )
        assert len(chunks) == 3
        assert all(chunk.id == "completion-test" for chunk in chunks)
    else:
        assert result.id == "completion-test"
    assert _usage_attributes(span_exporter) == expected
    metrics = metric_reader.get_metrics_data()
    assert metrics is not None
    token_points = [
        point
        for resource in metrics.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == GEN_AI_CLIENT_TOKEN_USAGE
        for point in metric.data.data_points
    ]
    assert len(token_points) == (2 if usage else 0)
    assert {
        point.attributes[GEN_AI_TOKEN_TYPE]: (point.count, point.sum)
        for point in token_points
    } == ({"input": (1, 100), "output": (1, 20)} if usage else {})


@pytest.mark.parametrize("error_source", ["sdk", "caller"])
@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(_CACHE_USAGE, _CACHE_ATTRIBUTES, id="cache"),
        pytest.param(_AUDIO_USAGE, _AUDIO_ATTRIBUTES, id="audio"),
        pytest.param(
            _IMAGE_OUTPUT_USAGE, _IMAGE_OUTPUT_ATTRIBUTES, id="image-output"
        ),
        pytest.param(_OPENAI_USAGE, _OPENAI_ATTRIBUTES, id="openai-details"),
    ],
)
@pytest.mark.asyncio
async def test_stream_errors_preserve_detailed_usage(
    monkeypatch: pytest.MonkeyPatch,
    instrument_portkey: object,
    span_exporter: InMemorySpanExporter,
    completion_api: str,
    asynchronous: bool,
    error_source: str,
    usage: dict[str, object],
    expected: Mapping[str, int],
) -> None:
    error = ConnectionError("stream failed")
    _mock_http(monkeypatch, [usage], error if error_source == "sdk" else None)
    client_type = (
        getattr(portkey_ai, "AsyncPortkey")
        if asynchronous
        else portkey_ai.Portkey
    )
    client = client_type(
        api_key="test-key",
        provider="openai",
        base_url="https://portkey.test/v1",
    )
    if completion_api == "chat":
        stream = client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )
    else:
        stream = client.prompts.completions.create(
            prompt_id="test-prompt", variables={}, stream=True
        )
    if asynchronous:
        stream = await stream

    assert not span_exporter.get_finished_spans()
    with pytest.raises(ConnectionError) as raised:
        if asynchronous:
            async with stream:
                async for _ in stream:
                    if error_source == "caller":
                        raise error
        else:
            with stream:
                for _ in stream:
                    if error_source == "caller":
                        raise error

    assert raised.value is error
    assert _usage_attributes(span_exporter) == expected
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ERROR_TYPE] == "ConnectionError"


@pytest.mark.parametrize("invalid", [None, -1, True, False, "12", 1.5, {}, []])
def test_invalid_detailed_counts_are_omitted(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    invalid: object,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        set_usage_properties(
            invocation,
            {
                **_TOTALS,
                "cache_creation_input_tokens": invalid,
                "cache_read_input_tokens": invalid,
                "prompt_tokens_details": {
                    "cache_write_tokens": invalid,
                    "cached_tokens": invalid,
                    "text_tokens": invalid,
                    "image_tokens": invalid,
                    "audio_tokens": invalid,
                },
                "completion_tokens_details": {
                    "text_tokens": invalid,
                    "image_tokens": invalid,
                    "audio_tokens": invalid,
                    "reasoning_tokens": invalid,
                },
            },
        )
    assert _usage_attributes(span_exporter) == _AGGREGATES


@pytest.mark.parametrize(
    "invalid",
    [None, -1, True, False, "12", 1.5, float("nan"), float("inf"), {}, []],
)
@pytest.mark.parametrize("previous_counts", [False, True])
@pytest.mark.parametrize("as_object", [False, True])
def test_invalid_aggregate_counts_do_not_replace_valid_usage(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    invalid: object,
    previous_counts: bool,
    as_object: bool,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        if previous_counts:
            set_usage_properties(invocation, _TOTALS)
        usage = {
            **_OPENAI_USAGE,
            "prompt_tokens": invalid,
            "completion_tokens": invalid,
        }
        set_usage_properties(
            invocation, SimpleNamespace(**usage) if as_object else usage
        )
        assert invocation.input_tokens == (100 if previous_counts else None)
        assert invocation.output_tokens == (20 if previous_counts else None)

    expected = dict(_OPENAI_ATTRIBUTES)
    if not previous_counts:
        expected.pop("gen_ai.usage.input_tokens")
        expected.pop("gen_ai.usage.output_tokens")
    assert _usage_attributes(span_exporter) == expected
    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET


def test_zero_aggregate_counts_replace_previous_totals(
    tracer_provider: TracerProvider, span_exporter: InMemorySpanExporter
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        set_usage_properties(invocation, _TOTALS)
        set_usage_properties(
            invocation, {"prompt_tokens": 0, "completion_tokens": 0}
        )
        assert invocation.input_tokens == 0
        assert invocation.output_tokens == 0
    assert _usage_attributes(span_exporter) == {
        "gen_ai.usage.input_tokens": 0,
        "gen_ai.usage.output_tokens": 0,
    }


def test_partial_usage_snapshots_preserve_previous_counts(
    tracer_provider: TracerProvider, span_exporter: InMemorySpanExporter
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        set_usage_properties(
            invocation,
            SimpleNamespace(
                **_TOTALS,
                cache_creation_input_tokens=30,
                cache_read_input_tokens=40,
                prompt_tokens_details=SimpleNamespace(
                    text_tokens=50, image_tokens=25, audio_tokens=25
                ),
                completion_tokens_details=SimpleNamespace(
                    text_tokens=8,
                    image_tokens=7,
                    audio_tokens=5,
                    reasoning_tokens=7,
                ),
            ),
        )
        set_usage_properties(
            invocation,
            {
                "prompt_tokens": 110,
                "cache_creation_input_tokens": 35,
                "prompt_tokens_details": {"cached_tokens": 45},
                "completion_tokens_details": {"audio_tokens": -1},
            },
        )
        set_usage_properties(invocation, None)
        set_usage_properties(invocation, {})

    assert _usage_attributes(span_exporter) == {
        **_OPENAI_ATTRIBUTES,
        "gen_ai.usage.text.output_tokens": 8,
        "gen_ai.usage.image.output_tokens": 7,
        "gen_ai.usage.input_tokens": 110,
        "gen_ai.usage.cache_write.input_tokens": 35,
        "gen_ai.usage.cache_read.input_tokens": 45,
    }


def test_zero_details_replace_previous_counts_without_cache_fallback(
    tracer_provider: TracerProvider, span_exporter: InMemorySpanExporter
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        set_usage_properties(invocation, _IMAGE_OUTPUT_USAGE)
        set_usage_properties(invocation, _OPENAI_USAGE)
        set_usage_properties(
            invocation,
            {
                "cache_creation_input_tokens": 30,
                "cache_read_input_tokens": 40,
                "prompt_tokens_details": {
                    "cache_write_tokens": 0,
                    "cached_tokens": 0,
                    "text_tokens": 0,
                    "image_tokens": 0,
                    "audio_tokens": 0,
                },
                "completion_tokens_details": {
                    "text_tokens": 0,
                    "image_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                },
            },
        )
    assert _usage_attributes(span_exporter) == _AGGREGATES


@pytest.mark.parametrize(
    "later_count,expected",
    [
        (None, 7),
        (-1, 7),
        (True, 7),
        ("12", 7),
        (1.5, 7),
        ({}, 7),
        ([], 7),
        (0, 0),
        (9, 9),
    ],
)
def test_reasoning_usage_updates_preserve_valid_counts(
    tracer_provider: TracerProvider,
    span_exporter: InMemorySpanExporter,
    later_count: object,
    expected: int,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    with handler.inference(
        "portkey", request_model="test-model"
    ) as invocation:
        set_usage_properties(invocation, _OPENAI_USAGE)
        set_usage_properties(
            invocation,
            {"completion_tokens_details": {"reasoning_tokens": later_count}},
        )
        assert invocation.thinking_tokens == expected
    expected_attributes = dict(_OPENAI_ATTRIBUTES)
    if expected:
        expected_attributes["gen_ai.usage.reasoning.output_tokens"] = expected
    else:
        expected_attributes.pop("gen_ai.usage.reasoning.output_tokens")
    assert _usage_attributes(span_exporter) == expected_attributes
