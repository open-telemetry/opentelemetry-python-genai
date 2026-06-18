# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import inspect
import json
from importlib import import_module

import pytest
from openai import (
    APIConnectionError,
    AsyncOpenAI,
    BadRequestError,
    NotFoundError,
)

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.util.genai.utils import is_experimental_mode

from .test_utils import (
    DEFAULT_MODEL,
    USER_ONLY_EXPECTED_INPUT_MESSAGES,
    USER_ONLY_PROMPT,
    assert_all_attributes,
    assert_cache_attributes,
    assert_messages_attribute,
    format_simple_expected_output_message,
    get_responses_weather_tool_definition,
)

try:
    # Responses is not available in the oldest supported OpenAI SDK, so keep
    # this import guarded. Pylint runs against the oldest dependency set and
    # cannot resolve this optional module there.
    _responses_module = import_module("openai.resources.responses.responses")
    HAS_RESPONSES_API = hasattr(_responses_module, "AsyncResponses")
    _create_params = set(
        inspect.signature(_responses_module.AsyncResponses.create).parameters
    )
    _has_tools_param = "tools" in _create_params
    _has_reasoning_param = "reasoning" in _create_params
except ImportError:
    HAS_RESPONSES_API = False
    _has_tools_param = False
    _has_reasoning_param = False


pytestmark = pytest.mark.skipif(
    not HAS_RESPONSES_API, reason="Responses API requires a newer openai SDK"
)

SYSTEM_INSTRUCTIONS = "You are a helpful assistant."
EXPECTED_SYSTEM_INSTRUCTIONS = [
    {
        "type": "text",
        "content": SYSTEM_INSTRUCTIONS,
    }
]
INVALID_MODEL = "this-model-does-not-exist"
REASONING_MODEL = "gpt-5.4"
REASONING_PROMPT = """
Write a bash script that takes a matrix represented as a string with
format '[1,2],[3,4],[5,6]' and prints the transpose in the same format.
"""


def _skip_if_not_latest():
    if not is_experimental_mode():
        pytest.skip(
            "Responses create instrumentation only supports the latest experimental semconv path"
        )


async def _collect_completed_response(stream):
    response = None
    async for event in stream:
        if event.type == "response.completed":
            response = event.response
    assert response is not None
    return response


def _load_span_messages(span, attribute):
    value = span.attributes.get(attribute)
    assert value is not None
    return json.loads(value)


def _assert_response_content(span, response, log_exporter):
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        USER_ONLY_EXPECTED_INPUT_MESSAGES,
    )
    assert (
        json.loads(span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS])
        == EXPECTED_SYSTEM_INSTRUCTIONS
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_simple_expected_output_message(response.output_text),
    )
    assert len(log_exporter.get_finished_logs()) == 0


def _assert_request_attrs(
    span,
    *,
    temperature=None,
    top_p=None,
    max_tokens=None,
    output_type=None,
):
    if temperature is not None:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE]
            == temperature
        )
    if top_p is not None:
        assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == top_p
    if max_tokens is not None:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS]
            == max_tokens
        )
    if output_type is not None:
        assert (
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_TYPE] == output_type
        )


def test_async_responses_uninstrument_removes_patching(
    span_exporter, tracer_provider, logger_provider, meter_provider
):
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    )
    instrumentor.uninstrument()

    assert len(span_exporter.get_finished_spans()) == 0


@pytest.mark.asyncio()
async def test_async_responses_create_basic(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_basic[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=False,
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response_service_tier=getattr(response, "service_tier", None),
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.asyncio()
async def test_async_responses_create_captures_content(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_with_content,
    vcr,
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_captures_content[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=False,
            text={"format": {"type": "text"}},
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response_service_tier=getattr(response, "service_tier", None),
    )
    _assert_response_content(span, response, log_exporter)


@pytest.mark.asyncio()
async def test_async_responses_create_with_all_params(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_with_all_params[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            max_output_tokens=50,
            temperature=0.7,
            top_p=0.9,
            service_tier="default",
            text={"format": {"type": "text"}},
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        request_service_tier="default",
        response_service_tier=getattr(response, "service_tier", None),
    )
    _assert_request_attrs(
        span,
        temperature=0.7,
        top_p=0.9,
        max_tokens=50,
        output_type="text",
    )


@pytest.mark.asyncio()
async def test_async_responses_create_token_usage(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_token_usage[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input="Count to 5.",
        )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response.usage.input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )


@pytest.mark.asyncio()
async def test_async_responses_create_aggregates_cache_tokens(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_aggregates_cache_tokens[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
        )

    (span,) = span_exporter.get_finished_spans()
    assert_cache_attributes(span, response.usage)


@pytest.mark.asyncio()
async def test_async_responses_create_stop_reason(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_stop_reason[content_mode0].yaml"
    ):
        await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input="Say hi.",
        )

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


@pytest.mark.asyncio()
async def test_async_responses_create_connection_error(
    span_exporter, instrument_no_content
):
    _skip_if_not_latest()

    client = AsyncOpenAI(base_url="http://localhost:4242")

    with pytest.raises(APIConnectionError):
        await client.responses.create(
            model=DEFAULT_MODEL,
            input="Hello",
            timeout=0.1,
        )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[ServerAttributes.SERVER_ADDRESS] == "localhost"
    assert span.attributes[ServerAttributes.SERVER_PORT] == 4242
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "APIConnectionError"


@pytest.mark.asyncio()
async def test_async_responses_create_api_error(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_api_error[content_mode0].yaml"
    ):
        with pytest.raises((BadRequestError, NotFoundError)) as exc_info:
            await async_openai_client.responses.create(
                model=INVALID_MODEL,
                input="Hello",
            )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == INVALID_MODEL
    )
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE]
        == type(exc_info.value).__name__
    )


@pytest.mark.asyncio()
async def test_async_responses_create_streaming(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            service_tier="default",
            stream=True,
        )
        response = await _collect_completed_response(stream)

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        request_service_tier="default",
        response_service_tier=getattr(response, "service_tier", None),
    )


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_aggregates_cache_tokens(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming_aggregates_cache_tokens[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=True,
        )
        response = await _collect_completed_response(stream)

    (span,) = span_exporter.get_finished_spans()
    assert_cache_attributes(span, response.usage)


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_captures_content(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_with_content,
    vcr,
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming_captures_content[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=True,
        )
        response = await _collect_completed_response(stream)

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response_service_tier=getattr(response, "service_tier", None),
    )
    _assert_response_content(span, response, log_exporter)


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_iteration(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming_iteration[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input="Say hi.",
            stream=True,
        )
        events = [event async for event in stream]

    assert len(events) > 0

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert GenAIAttributes.GEN_AI_RESPONSE_ID in span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS in span.attributes
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS in span.attributes


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_delegates_response_attribute(
    async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming_delegates_response_attribute[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input="Say hi.",
            stream=True,
        )

        assert stream.response is not None
        assert stream.response.status_code == 200
        assert stream.response.headers.get("x-request-id") is not None
        await stream.close()


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_connection_error(
    span_exporter, instrument_no_content
):
    _skip_if_not_latest()

    client = AsyncOpenAI(base_url="http://localhost:4242")

    with pytest.raises(APIConnectionError):
        await client.responses.create(
            model=DEFAULT_MODEL,
            input="Hello",
            stream=True,
            timeout=0.1,
        )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "APIConnectionError"


@pytest.mark.asyncio()
async def test_async_responses_stream_wrapper_finalize_idempotent(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_stream_wrapper_finalize_idempotent[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=True,
        )

        response = await _collect_completed_response(stream)
        await stream.close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert_all_attributes(
        spans[0],
        DEFAULT_MODEL,
        True,
        response.id,
        response.model,
        response.usage.input_tokens,
        response.usage.output_tokens,
        response_service_tier=getattr(response, "service_tier", None),
    )


@pytest.mark.asyncio()
async def test_async_responses_create_stream_propagation_error(
    span_exporter, async_openai_client, instrument_no_content, monkeypatch, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_stream_propagation_error[content_mode0].yaml"
    ):
        stream = await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=True,
        )

        class ErrorInjectingStreamDelegate:
            def __init__(self, inner):
                self._inner = inner
                self._count = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._count == 1:
                    raise ConnectionError("connection reset during stream")
                self._count += 1
                return await self._inner.__anext__()

            async def close(self):
                return await self._inner.close()

            def __getattr__(self, name):
                return getattr(self._inner, name)

        monkeypatch.setattr(
            stream, "stream", ErrorInjectingStreamDelegate(stream.stream)
        )

        with pytest.raises(
            ConnectionError, match="connection reset during stream"
        ):
            async with stream:
                async for _ in stream:
                    pass

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


@pytest.mark.asyncio()
async def test_async_responses_create_streaming_user_exception(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_streaming_user_exception[content_mode0].yaml"
    ):
        with pytest.raises(ValueError, match="User raised exception"):
            async with await async_openai_client.responses.create(
                model=DEFAULT_MODEL,
                instructions=SYSTEM_INSTRUCTIONS,
                input=USER_ONLY_PROMPT[0]["content"],
                stream=True,
            ) as stream:
                async for _ in stream:
                    raise ValueError("User raised exception")

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio()
@pytest.mark.skipif(
    not _has_tools_param,
    reason="openai SDK too old to support 'tools' parameter on Responses.create",
)
async def test_async_responses_create_captures_tool_call_content(
    span_exporter, async_openai_client, instrument_with_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_captures_tool_call_content[content_mode0].yaml"
    ):
        await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            input="What's the weather in Seattle right now?",
            tools=[get_responses_weather_tool_definition()],
            tool_choice={"type": "function", "name": "get_current_weather"},
        )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == DEFAULT_MODEL
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_calls",
    )

    input_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    assert input_messages[0]["role"] == "user"

    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    tool_call_parts = [
        part
        for message in output_messages
        for part in message.get("parts", [])
        if part.get("type") == "tool_call"
    ]
    assert len(tool_call_parts) > 0
    assert tool_call_parts[0]["name"] == "get_current_weather"
    assert "arguments" in tool_call_parts[0]


@pytest.mark.asyncio()
@pytest.mark.skipif(
    not _has_reasoning_param,
    reason=(
        "openai SDK too old to support 'reasoning' parameter on Responses.create"
    ),
)
async def test_async_responses_create_reports_reasoning_tokens(
    span_exporter, async_openai_client, instrument_with_content, vcr
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_reports_reasoning_tokens[content_mode0].yaml"
    ):
        response = await async_openai_client.responses.create(
            model=REASONING_MODEL,
            reasoning={"effort": "low"},
            input=[
                {
                    "role": "user",
                    "content": REASONING_PROMPT,
                }
            ],
            max_output_tokens=1000,
            timeout=30.0,
        )

    reasoning_tokens = getattr(
        getattr(response.usage, "output_tokens_details", None),
        "reasoning_tokens",
        None,
    )

    assert reasoning_tokens is not None
    assert reasoning_tokens > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    (span,) = spans
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == (
        REASONING_MODEL
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS]
        == response.usage.input_tokens
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS]
        == response.usage.output_tokens
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )
    assert len(output_messages) > 0


@pytest.mark.asyncio()
async def test_async_responses_create_with_content_span_unsampled(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_with_content_unsampled,
    vcr,
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_with_content_span_unsampled[content_mode0].yaml"
    ):
        await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=False,
        )

    assert len(span_exporter.get_finished_spans()) == 0
    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.asyncio()
async def test_async_responses_create_with_content_shapes(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_with_content,
    vcr,
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_with_content_shapes[content_mode0].yaml"
    ):
        await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=False,
        )

    (span,) = span_exporter.get_finished_spans()
    input_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_INPUT_MESSAGES
    )
    output_messages = _load_span_messages(
        span, GenAIAttributes.GEN_AI_OUTPUT_MESSAGES
    )

    assert input_messages[0]["role"] == "user"
    assert input_messages[0]["parts"][0]["type"] == "text"
    assert output_messages[0]["role"] == "assistant"
    assert output_messages[0]["parts"][0]["type"] == "text"
    assert len(log_exporter.get_finished_logs()) == 0


@pytest.mark.asyncio()
async def test_async_responses_create_event_only_no_content_in_span(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_event_only,
    vcr,
):
    _skip_if_not_latest()

    with vcr.use_cassette(
        "test_async_responses_create_event_only_no_content_in_span.yaml"
    ):
        await async_openai_client.responses.create(
            model=DEFAULT_MODEL,
            instructions=SYSTEM_INSTRUCTIONS,
            input=USER_ONLY_PROMPT[0]["content"],
            stream=False,
        )

    (span,) = span_exporter.get_finished_spans()
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes

    logs = log_exporter.get_finished_logs()
    assert len(logs) == 1
    assert (
        logs[0].log_record.event_name
        == "gen_ai.client.inference.operation.details"
    )
