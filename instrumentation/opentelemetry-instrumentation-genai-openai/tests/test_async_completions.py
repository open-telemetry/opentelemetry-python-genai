# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# Each test replays the cassette of its sync twin in test_completions.py:
# the async client sends the same request.

import pytest
from openai import APIConnectionError, AsyncOpenAI, NotFoundError

from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)

from .test_utils import (
    COMPLETION_MODEL,
    COMPLETION_PROMPT,
    TEXT_COMPLETION_OPERATION_NAME,
    USER_ONLY_EXPECTED_INPUT_MESSAGES,
    assert_all_attributes,
    assert_inference_event_without_content,
    assert_messages_attribute,
    format_text_output_messages,
)


@pytest.mark.asyncio()
async def test_async_completion_with_content(
    span_exporter, async_openai_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_completion_with_content.yaml"):
        response = await async_openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
        )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        response.id,
        response.model,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        response.choices[0].finish_reason,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        USER_ONLY_EXPECTED_INPUT_MESSAGES,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages(response.choices[0].text),
    )


@pytest.mark.asyncio()
async def test_async_completion_no_content(
    span_exporter,
    log_exporter,
    async_openai_client,
    instrument_no_content,
    vcr,
):
    with vcr.use_cassette("test_completion_with_content.yaml"):
        await async_openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
        )

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == TEXT_COMPLETION_OPERATION_NAME
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert_inference_event_without_content(log_exporter.get_finished_logs())


@pytest.mark.asyncio()
async def test_async_completion_404(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    llm_model_value = "this-model-does-not-exist"

    with vcr.use_cassette("test_completion_404.yaml"):
        with pytest.raises(NotFoundError):
            await async_openai_client.completions.create(
                model=llm_model_value,
                prompt=COMPLETION_PROMPT,
            )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        llm_model_value,
        True,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE] == "openai.NotFoundError"
    )


@pytest.mark.asyncio()
async def test_async_completion_bad_endpoint(
    span_exporter, instrument_no_content, vcr
):
    client = AsyncOpenAI(base_url="http://localhost:4242")

    with vcr.use_cassette("test_async_completion_bad_endpoint.yaml"):
        with pytest.raises(APIConnectionError):
            await client.completions.create(
                model=COMPLETION_MODEL,
                prompt=COMPLETION_PROMPT,
                timeout=0.1,
            )

    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        server_address="localhost",
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[ServerAttributes.SERVER_PORT] == 4242
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE]
        == "openai.APIConnectionError"
    )


@pytest.mark.asyncio()
async def test_async_completion_streaming(
    span_exporter, async_openai_client, instrument_with_content, vcr
):
    text = ""
    finish_reason = None
    last_chunk = None

    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = await async_openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            for choice in chunk.choices:
                text += choice.text
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
            last_chunk = chunk

    assert last_chunk.usage is not None
    (span,) = span_exporter.get_finished_spans()
    assert_all_attributes(
        span,
        COMPLETION_MODEL,
        True,
        last_chunk.id,
        last_chunk.model,
        last_chunk.usage.prompt_tokens,
        last_chunk.usage.completion_tokens,
        operation_name=TEXT_COMPLETION_OPERATION_NAME,
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        finish_reason,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES],
        USER_ONLY_EXPECTED_INPUT_MESSAGES,
    )
    assert_messages_attribute(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
        format_text_output_messages(text),
    )


@pytest.mark.asyncio()
async def test_async_completion_streaming_no_content(
    span_exporter, async_openai_client, instrument_no_content, vcr
):
    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = await async_openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for _ in response:
            pass

    (span,) = span_exporter.get_finished_spans()
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == TEXT_COMPLETION_OPERATION_NAME
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.asyncio()
async def test_async_completion_streaming_user_exception_propagates(
    span_exporter, async_openai_client, instrument_with_content, vcr
):
    with vcr.use_cassette("test_completion_streaming.yaml"):
        response = await async_openai_client.completions.create(
            model=COMPLETION_MODEL,
            prompt=COMPLETION_PROMPT,
            max_tokens=16,
            stream=True,
            stream_options={"include_usage": True},
        )
        with pytest.raises(RuntimeError, match="user failure"):
            async with response:
                async for _ in response:
                    raise RuntimeError("user failure")

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"
