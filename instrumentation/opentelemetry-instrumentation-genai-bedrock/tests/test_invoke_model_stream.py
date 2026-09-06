# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock InvokeModelWithResponseStream API instrumentation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock.patch import (
    _handle_invoke_model,
)
from opentelemetry.instrumentation.genai.bedrock.stream import (
    BedrockInvokeModelStreamWrapper,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def test_stream_wrapper_anthropic_messages(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="anthropic.claude-3-sonnet-20240229-v1:0",
        operation_name="chat",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "usage": {"input_tokens": 14},
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": "Let me think...",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {
                            "type": "text_delta",
                            "text": "Hello world!",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 20},
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    stream_events = list(wrapper)
    assert len(stream_events) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat anthropic.claude-3-sonnet-20240229-v1:0"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 14
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    parts = output_msgs[0]["parts"]
    assert len(parts) == 2
    assert parts[0]["type"] == "reasoning"
    assert parts[0]["content"] == "Let me think..."
    assert parts[1]["type"] == "text"
    assert parts[1]["content"] == "Hello world!"


def test_stream_wrapper_titan_with_metrics(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "Here is ",
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "the response.",
                        "completionReason": "FINISH",
                        "amazon-bedrock-invocationMetrics": {
                            "inputTokenCount": 6,
                            "outputTokenCount": 12,
                        },
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    stream_events = list(wrapper)
    assert len(stream_events) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.titan-text-lite-v1"
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 6
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["parts"][0]["content"] == "Here is the response."


def test_stream_wrapper_llama_and_mistral(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="meta.llama3-8b-instruct-v1:0",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {"generation": "Llama says hi", "stop_reason": "stop"}
                ).encode("utf-8")
            }
        }
    ]
    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert output_msgs[0]["parts"][0]["content"] == "Llama says hi"


def test_stream_wrapper_no_content(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "secret text",
                        "completionReason": "FINISH",
                        "amazon-bedrock-invocationMetrics": {
                            "inputTokenCount": 5,
                            "outputTokenCount": 10,
                        },
                    }
                ).encode("utf-8")
            }
        }
    ]
    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=False,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_stream_wrapper_caller_side_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps({"outputText": "chunk 1"}).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
    )

    with pytest.raises(RuntimeError, match="caller exploded"):
        with wrapper as stream:
            for _ in stream:
                raise RuntimeError("caller exploded")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


def test_stream_wrapper_stream_side_error(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="amazon.titan-text-lite-v1",
    )

    class FailingEventStream:
        def __iter__(self):
            yield {"chunk": {"bytes": b'{"outputText": "chunk"}'}}
            raise ConnectionError("connection reset")

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=FailingEventStream(),  # type: ignore[arg-type]
        invocation=invocation,
    )

    with pytest.raises(ConnectionError, match="connection reset"):
        list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_handle_invoke_model_streaming_integration(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "outputText": "Streamed text.",
                        "completionReason": "FINISH",
                    }
                ).encode("utf-8")
            }
        }
    ]
    mock_instance = MagicMock()
    mock_instance.meta.endpoint_url = (
        "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    mock_wrapped = MagicMock(return_value={"body": events})

    api_params = {
        "modelId": "amazon.titan-text-lite-v1",
        "body": json.dumps({"inputText": "hello"}),
    }

    response = _handle_invoke_model(
        mock_wrapped,
        mock_instance,
        ("InvokeModelWithResponseStream", api_params),
        {},
        api_params,
        handler,
        is_stream=True,
    )

    assert isinstance(response["body"], BedrockInvokeModelStreamWrapper)
    list(response["body"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "chat amazon.titan-text-lite-v1"


def test_stream_wrapper_anthropic_with_cache_tokens(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "usage": {
                                "input_tokens": 50,
                                "cache_read_input_tokens": 30,
                                "cache_creation_input_tokens": 10,
                            },
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "text_delta",
                            "text": "Hello world!",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 5},
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 50
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 30
    )
    assert (
        span.attributes[
            GenAIAttributes.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
        ]
        == 10
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_stream_wrapper_anthropic_tool_calls(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_AND_EVENT"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="anthropic.claude-3-sonnet-20240229-v1:0",
        operation_name="chat",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "usage": {"input_tokens": 30},
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_01T1xnvuzHG2DhStub123",
                            "name": "get_stock_price",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"ticker":',
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": ' "AAPL"}',
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 25},
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=True,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 30
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 25

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    parts = output_msgs[0]["parts"]
    assert len(parts) == 1
    assert parts[0]["type"] == "tool_call"
    assert parts[0]["id"] == "toolu_01T1xnvuzHG2DhStub123"
    assert parts[0]["name"] == "get_stock_price"
    assert parts[0]["arguments"] == {"ticker": "AAPL"}


def test_stream_wrapper_anthropic_tool_calls_no_content(
    tracer_provider,
    span_exporter,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(
        provider="aws.bedrock",
        request_model="anthropic.claude-3-sonnet-20240229-v1:0",
        operation_name="chat",
    )
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_01T1xnvuzHG2DhStub123",
                            "name": "get_stock_price",
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"ticker": "AAPL"}',
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                    }
                ).encode("utf-8")
            }
        },
    ]

    wrapper = BedrockInvokeModelStreamWrapper(
        stream=events,  # type: ignore[arg-type]
        invocation=invocation,
        capture_content=False,
    )
    list(wrapper)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert not wrapper._self_tool_blocks
    assert not wrapper._self_text_blocks


def test_invoke_model_with_response_stream_end_to_end_stubber(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    stubber._validate_response = lambda *a, **kw: None

    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    events = [
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "role": "assistant",
                            "usage": {
                                "input_tokens": 12,
                                "cache_read_input_tokens": 8,
                            },
                        },
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Hello "},
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "world!"},
                    }
                ).encode("utf-8")
            }
        },
        {
            "chunk": {
                "bytes": json.dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 6},
                    }
                ).encode("utf-8")
            }
        },
    ]

    stubber.add_response(
        "invoke_model_with_response_stream",
        service_response={
            "contentType": "application/json",
            "body": events,
        },
        expected_params={
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model_with_response_stream(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(request_body),
        )

        assert isinstance(response["body"], BedrockInvokeModelStreamWrapper)
        stream_events = list(response["body"])
        assert len(stream_events) == 4

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat anthropic.claude-3-sonnet-20240229-v1:0"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "bedrock-runtime.us-east-1.amazonaws.com"
    )
    assert span.attributes[ServerAttributes.SERVER_PORT] == 443
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STREAM] is True
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 6
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 8
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["parts"][0]["content"] == "Hello world!"
