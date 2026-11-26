# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy ``completions.create`` served by an in-process mock transport.

Covers what the recorded tests in ``test_completions.py`` don't: several
prompts, token prompts, ``n > 1``, raw responses, a stream that fails
mid-read, and uninstrument.
"""

from __future__ import annotations

import asyncio
import gc
import json

import openai
import openai._base_client as base_client
import pytest
from openai import NotFoundError, Stream
from openai.types import Completion
from pydantic import BaseModel

from opentelemetry import trace
from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    openai_attributes as OpenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind

from .test_utils import (
    COMPLETION_MODEL,
    TEXT_COMPLETION_OPERATION_NAME,
    assert_messages_attribute,
    format_text_output_messages,
)

# Newer openai releases vendor httpx as httpx2.
httpx = getattr(base_client, "httpx2", None) or base_client.httpx

_PROMPTS = ["Say this is a test", "Say hi"]
_RESPONSE_MODEL = "gpt-3.5-turbo-instruct:20230824-v2"
_USAGE = {"prompt_tokens": 9, "completion_tokens": 7, "total_tokens": 16}


def _completion(choices, usage=None):
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "created": 1,
        "model": _RESPONSE_MODEL,
        "system_fingerprint": "fp_1",
        "choices": [
            {
                "index": index,
                "text": text,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
            for index, text, finish_reason in choices
        ],
        "usage": usage,
    }


_ANSWER = _completion(
    [(0, "This is a test.", "stop"), (1, "Hi", "length")], _USAGE
)
_CHUNKS = [
    _completion([(0, "This is", None)]),
    _completion([(1, "Hi", None)]),
    _completion([(0, " a test.", "stop")]),
    _completion([(1, "", "length")]),
    _completion([], _USAGE),
]
_EXPECTED_OUTPUT_MESSAGES = format_text_output_messages(
    "This is a test.", "Hi"
)


def _expected_input_messages(prompts):
    return [
        {"role": "user", "parts": [{"type": "text", "content": prompt}]}
        for prompt in prompts
    ]


def _sse_event(chunk) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def _json_handler(body, status=200):
    return lambda request: httpx.Response(status, json=body)


def _sse_handler(chunks):
    body = (
        b"".join(_sse_event(chunk) for chunk in chunks) + b"data: [DONE]\n\n"
    )
    return lambda request: httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=body
    )


class _FailingStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Response body that sends one SSE event, then fails the read."""

    def __init__(self, first: bytes, error: BaseException) -> None:
        self._first = first
        self._error = error

    def __iter__(self):
        yield self._first
        raise self._error

    async def __aiter__(self):
        yield self._first
        raise self._error


def _failing_sse_handler(error: BaseException):
    return lambda request: httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_FailingStream(_sse_event(_CHUNKS[0]), error),
    )


def _client(handler):
    return openai.OpenAI(
        api_key="x",
        base_url="http://mock/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _async_client(handler):
    return openai.AsyncOpenAI(
        api_key="x",
        base_url="http://mock/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _create(asynchronous, handler, **kwargs):
    if not asynchronous:
        return _client(handler).completions.create(**kwargs)

    async def create():
        return await _async_client(handler).completions.create(**kwargs)

    return asyncio.run(create())


def _assert_attributes(span, expected):
    for key, value in expected.items():
        assert span.attributes[key] == value, key
        assert type(span.attributes[key]) is type(value), key


@pytest.fixture(autouse=True)
def fixture_vcr():
    """Requests are served by a mock transport, so no cassettes here."""
    yield


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completion_records_text_completion_span(
    span_exporter, metric_reader, instrument_with_content, asynchronous
):
    response = _create(
        asynchronous,
        _json_handler(_ANSWER),
        model=COMPLETION_MODEL,
        prompt=_PROMPTS,
        max_tokens=8,
        temperature=0.5,
        top_p=0.9,
        stop=["\n"],
        seed=7,
        frequency_penalty=0.1,
        presence_penalty=0.2,
    )

    assert isinstance(response, Completion)
    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"{TEXT_COMPLETION_OPERATION_NAME} {COMPLETION_MODEL}"
    assert span.kind is SpanKind.CLIENT
    _assert_attributes(
        span,
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: TEXT_COMPLETION_OPERATION_NAME,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: "openai",
            GenAIAttributes.GEN_AI_REQUEST_MODEL: COMPLETION_MODEL,
            GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS: 8,
            GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE: 0.5,
            GenAIAttributes.GEN_AI_REQUEST_TOP_P: 0.9,
            GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES: ("\n",),
            GenAIAttributes.GEN_AI_REQUEST_SEED: 7,
            GenAIAttributes.GEN_AI_REQUEST_FREQUENCY_PENALTY: 0.1,
            GenAIAttributes.GEN_AI_REQUEST_PRESENCE_PENALTY: 0.2,
            GenAIAttributes.GEN_AI_RESPONSE_ID: "cmpl-1",
            GenAIAttributes.GEN_AI_RESPONSE_MODEL: _RESPONSE_MODEL,
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS: ("stop", "length"),
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 9,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 7,
            OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT: "fp_1",
            ServerAttributes.SERVER_ADDRESS: "mock",
        },
    )
    assert GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT not in span.attributes
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        _expected_input_messages(_PROMPTS),
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        _EXPECTED_OUTPUT_MESSAGES,
    )

    (resource_metrics,) = metric_reader.get_metrics_data().resource_metrics
    metrics = {
        metric.name: metric
        for metric in resource_metrics.scope_metrics[0].metrics
    }
    for name in (
        gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION,
        gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE,
    ):
        for point in metrics[name].data.data_points:
            assert (
                point.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
                == TEXT_COMPLETION_OPERATION_NAME
            )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completion_no_content(
    span_exporter, instrument_no_content, asynchronous
):
    _create(
        asynchronous,
        _json_handler(_ANSWER),
        model=COMPLETION_MODEL,
        prompt=_PROMPTS,
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
        "length",
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.parametrize("prompt", [[1212, 318], [[1212, 318], [40, 1101]]])
def test_completion_token_prompt_records_no_input_messages(
    span_exporter, instrument_with_content, prompt
):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=_ANSWER)

    _client(handler).completions.create(model=COMPLETION_MODEL, prompt=prompt)

    assert requests[0]["prompt"] == prompt
    (span,) = span_exporter.get_finished_spans()
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        _EXPECTED_OUTPUT_MESSAGES,
    )


def _assert_streamed_span(span_exporter):
    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"{TEXT_COMPLETION_OPERATION_NAME} {COMPLETION_MODEL}"
    _assert_attributes(
        span,
        {
            GenAIAttributes.GEN_AI_OPERATION_NAME: TEXT_COMPLETION_OPERATION_NAME,
            GenAIAttributes.GEN_AI_REQUEST_CHOICE_COUNT: 2,
            GenAIAttributes.GEN_AI_RESPONSE_ID: "cmpl-1",
            GenAIAttributes.GEN_AI_RESPONSE_MODEL: _RESPONSE_MODEL,
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS: ("stop", "length"),
            GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS: 9,
            GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS: 7,
            OpenAIAttributes.OPENAI_RESPONSE_SYSTEM_FINGERPRINT: "fp_1",
        },
    )
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        _expected_input_messages(_PROMPTS[:1]),
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        _EXPECTED_OUTPUT_MESSAGES,
    )


_STREAM_KWARGS = {
    "model": COMPLETION_MODEL,
    "prompt": _PROMPTS[0],
    "n": 2,
    "stream": True,
    "stream_options": {"include_usage": True},
}


def test_completion_streaming(span_exporter, instrument_with_content):
    stream = _client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    assert not trace.get_current_span().get_span_context().is_valid
    assert span_exporter.get_finished_spans() == ()
    assert [chunk.choices[0].text for chunk in stream if chunk.choices] == [
        "This is",
        "Hi",
        " a test.",
        "",
    ]
    _assert_streamed_span(span_exporter)


@pytest.mark.asyncio()
async def test_async_completion_streaming(
    span_exporter, instrument_with_content
):
    stream = await _async_client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    assert not trace.get_current_span().get_span_context().is_valid
    assert span_exporter.get_finished_spans() == ()
    assert len([chunk async for chunk in stream]) == len(_CHUNKS)
    _assert_streamed_span(span_exporter)


def test_completion_streaming_no_content(span_exporter, instrument_no_content):
    for _ in _client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    ):
        pass

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
        "length",
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


def test_completion_stream_error_mid_read(
    span_exporter, instrument_with_content
):
    error = ConnectionError("connection dropped")
    stream = _client(_failing_sse_handler(error)).completions.create(
        **_STREAM_KWARGS
    )

    assert next(stream).choices[0].text == "This is"
    with pytest.raises(ConnectionError) as raised:
        next(stream)

    assert raised.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio()
async def test_async_completion_stream_error_mid_read(
    span_exporter, instrument_with_content
):
    error = ConnectionError("connection dropped")
    stream = await _async_client(
        _failing_sse_handler(error)
    ).completions.create(**_STREAM_KWARGS)

    assert (await anext(stream)).choices[0].text == "This is"
    with pytest.raises(ConnectionError) as raised:
        await anext(stream)

    assert raised.value is error
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_completion_stream_closed_early(
    span_exporter, instrument_with_content
):
    stream = _client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    next(stream)
    stream.close()

    (span,) = span_exporter.get_finished_spans()
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages("This is"),
    )


@pytest.mark.asyncio()
async def test_async_completion_stream_closed_early(
    span_exporter, instrument_with_content
):
    stream = await _async_client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    await anext(stream)
    await stream.close()

    (span,) = span_exporter.get_finished_spans()
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert (
        GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS not in span.attributes
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages("This is"),
    )


def test_completion_stream_caller_error(
    span_exporter, instrument_with_content
):
    stream = _client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    with pytest.raises(RuntimeError, match="user failure"):
        with stream:
            next(stream)
            raise RuntimeError("user failure")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio()
async def test_async_completion_stream_caller_error(
    span_exporter, instrument_with_content
):
    stream = await _async_client(_sse_handler(_CHUNKS)).completions.create(
        **_STREAM_KWARGS
    )

    with pytest.raises(RuntimeError, match="user failure"):
        async with stream:
            await anext(stream)
            raise RuntimeError("user failure")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_completion_error_response(
    span_exporter, instrument_no_content, asynchronous
):
    error_body = {
        "error": {
            "message": "The model `missing` does not exist",
            "type": "invalid_request_error",
            "code": "model_not_found",
        }
    }

    with pytest.raises(NotFoundError):
        _create(
            asynchronous,
            _json_handler(error_body, status=404),
            model="missing",
            prompt=_PROMPTS[0],
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.name == f"{TEXT_COMPLETION_OPERATION_NAME} missing"
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE] == "openai.NotFoundError"
    )


def test_completion_with_raw_response(span_exporter, instrument_with_content):
    raw = _client(_json_handler(_ANSWER)).completions.with_raw_response.create(
        model=COMPLETION_MODEL, prompt=_PROMPTS
    )

    assert isinstance(raw.parse(), Completion)
    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
        "length",
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        _EXPECTED_OUTPUT_MESSAGES,
    )


def test_completion_with_raw_response_streaming(
    span_exporter, instrument_with_content
):
    raw = _client(_sse_handler(_CHUNKS)).completions.with_raw_response.create(
        **_STREAM_KWARGS
    )

    assert span_exporter.get_finished_spans() == ()
    for _ in raw.parse():
        pass

    _assert_streamed_span(span_exporter)


@pytest.mark.asyncio()
async def test_async_completion_with_streaming_response(
    span_exporter, instrument_with_content, recwarn
):
    client = _async_client(_json_handler(_ANSWER))
    async with client.completions.with_streaming_response.create(
        model=COMPLETION_MODEL, prompt=_PROMPTS
    ) as response:
        assert isinstance(await response.parse(), Completion)
    gc.collect()

    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]
    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == TEXT_COMPLETION_OPERATION_NAME
    )
    assert ErrorAttributes.ERROR_TYPE not in span.attributes


class _UnrelatedChunk(BaseModel):
    foo: str = "bar"


def test_completion_with_raw_response_streaming_unknown_chunk_type(
    span_exporter, instrument_with_content
):
    raw = _client(_sse_handler(_CHUNKS)).completions.with_raw_response.create(
        **_STREAM_KWARGS
    )

    chunks = list(raw.parse(to=Stream[_UnrelatedChunk]))

    assert len(chunks) == len(_CHUNKS)
    (span,) = span_exporter.get_finished_spans()
    assert ErrorAttributes.ERROR_TYPE not in span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in span.attributes


def test_uninstrument_stops_tracing_completions(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    client = _client(_json_handler(_ANSWER))
    with instrument(
        OpenAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        # The SDK caches with_raw_response, along with the wrapped create.
        client.completions.with_raw_response.create(
            model=COMPLETION_MODEL, prompt=_PROMPTS
        ).parse()
        assert len(span_exporter.get_finished_spans()) == 1

    span_exporter.clear()
    client.completions.create(model=COMPLETION_MODEL, prompt=_PROMPTS)
    client.completions.with_raw_response.create(
        model=COMPLETION_MODEL, prompt=_PROMPTS
    ).parse()
    assert len(span_exporter.get_finished_spans()) == 0
