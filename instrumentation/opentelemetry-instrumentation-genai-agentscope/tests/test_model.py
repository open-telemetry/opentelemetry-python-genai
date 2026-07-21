# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""AgentScope chat model instrumentation tests."""

from __future__ import annotations

import asyncio

import agentscope
import pytest
from agentscope.model import DashScopeChatModel

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ._test_helpers import assert_no_removed_telemetry


def _assert_chat_span_attributes(
    span,
    request_model: str,
    *,
    expect_input_messages: bool = False,
    expect_output_messages: bool = False,
) -> None:
    """Assert common chat model span attributes."""
    assert span.name.startswith("chat "), f"Unexpected span name: {span.name}"
    assert request_model in span.name

    # Standard OpenTelemetry operation-name attribute is always present, and
    # the removed gen_ai.span.kind attribute must never be set.
    assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
    assert "gen_ai.span.kind" not in span.attributes

    assert span.attributes["gen_ai.provider.name"] == "dashscope"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == request_model
    )

    if expect_input_messages:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES in span.attributes
    else:
        assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes

    if expect_output_messages:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES in span.attributes
    else:
        assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


async def _drain(response):
    try:
        is_stream = hasattr(response, "__aiter__")
    except (KeyError, AttributeError):
        is_stream = False
    if is_stream:
        result = []
        async for chunk in response:
            result.append(chunk)
        return result[-1] if result else response
    return response


@pytest.mark.vcr()
def test_model_call_basic(instrument_no_content, span_exporter, request):
    agentscope.init(project="test_basic")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )
    messages = [{"role": "user", "content": "Hello!"}]

    response = asyncio.run(_wrap(model, messages))
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(chat_spans[0], request_model="qwen-max")


async def _wrap(model, messages, **kwargs):
    response = await model(messages, **kwargs)
    return await _drain(response)


@pytest.mark.vcr()
def test_model_call_with_messages(
    instrument_no_content, span_exporter, request
):
    agentscope.init(project="test_messages")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1+1?"},
    ]

    response = asyncio.run(_wrap(model, messages))
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(chat_spans[0], request_model="qwen-max")


@pytest.mark.vcr()
async def test_model_call_async(instrument_no_content, span_exporter, request):
    agentscope.init(project="test_async")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )
    messages = [{"role": "user", "content": "Hello from async!"}]

    response = await _wrap(model, messages)
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(chat_spans[0], request_model="qwen-max")


@pytest.mark.vcr()
def test_model_call_streaming(instrument, span_exporter, request):
    agentscope.init(project="test_streaming")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=True,
    )
    messages = [{"role": "user", "content": "Count from 1 to 5"}]

    response = asyncio.run(_wrap(model, messages))
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(chat_spans[0], request_model="qwen-max")


@pytest.mark.vcr()
def test_model_call_with_parameters(instrument, span_exporter, request):
    agentscope.init(project="test_parameters")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )
    messages = [{"role": "user", "content": "Write a short poem"}]

    response = asyncio.run(
        _wrap(model, messages, temperature=0.8, top_p=0.9, max_tokens=100)
    )
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(chat_spans[0], request_model="qwen-max")


@pytest.mark.vcr()
def test_model_call_with_content_capture(
    instrument_with_content, span_exporter, request
):
    agentscope.init(project="test_content_capture")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
        stream=False,
    )
    messages = [{"role": "user", "content": "Say this is a test"}]

    response = asyncio.run(_wrap(model, messages))
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(
        chat_spans[0],
        request_model="qwen-max",
        expect_input_messages=True,
        expect_output_messages=True,
    )


@pytest.mark.vcr()
def test_model_call_no_content_capture(
    instrument_no_content, span_exporter, request
):
    agentscope.init(project="test_no_content_capture")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )
    messages = [{"role": "user", "content": "Say this is a test"}]

    response = asyncio.run(_wrap(model, messages))
    assert response is not None

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 1
    _assert_chat_span_attributes(
        chat_spans[0],
        request_model="qwen-max",
        expect_input_messages=False,
        expect_output_messages=False,
    )


@pytest.mark.vcr()
def test_model_call_multiple_sequential(instrument, span_exporter, request):
    agentscope.init(project="test_multiple")
    model = DashScopeChatModel(
        api_key=request.config.option.api_key,
        model_name="qwen-max",
    )

    async def call(content: str):
        return await _wrap(model, [{"role": "user", "content": content}])

    asyncio.run(call("First call"))
    asyncio.run(call("Second call"))
    asyncio.run(call("Third call"))

    spans = span_exporter.get_finished_spans()
    assert_no_removed_telemetry(spans)
    chat_spans = [s for s in spans if s.name.startswith("chat ")]
    assert len(chat_spans) >= 3
    for chat_span in chat_spans[:3]:
        _assert_chat_span_attributes(chat_span, request_model="qwen-max")
