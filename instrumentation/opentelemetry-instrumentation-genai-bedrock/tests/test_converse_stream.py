# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock converse_stream API instrumentation."""

from unittest import mock

import pytest
from botocore.eventstream import EventStream
from botocore.exceptions import ClientError, EventStreamError

from opentelemetry.instrumentation.genai.bedrock.stream import (
    BedrockConverseStreamWrapper,
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


@pytest.mark.vcr
def test_converse_stream_with_content(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    events = list(response["stream"])
    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
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
        == "amazon.nova-micro-v1:0"
    )
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in span.attributes
    assert GenAIAttributes.GEN_AI_RESPONSE_ID not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.8
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 1.0
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == (
        "|",
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        is not None
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "bedrock-runtime.us-east-1.amazonaws.com"
    )
    assert span.attributes[ServerAttributes.SERVER_PORT] == 443
    assert span.status.status_code == StatusCode.UNSET


@pytest.mark.vcr
def test_converse_stream_no_content(
    bedrock_client,
    instrument_no_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    events = list(response["stream"])
    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        is not None
    )


@pytest.mark.vcr
def test_converse_stream_with_content_tool_call(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "What is the weather in Seattle and San Francisco"
                        " today?"
                    )
                }
            ],
        }
    ]
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_current_weather",
                    "description": (
                        "Get the current weather in a given location."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The name of the city",
                                }
                            },
                            "required": ["location"],
                        }
                    },
                }
            }
        ]
    }

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        toolConfig=tool_config,
    )

    events = list(response["stream"])
    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )


@pytest.mark.vcr
def test_converse_stream_no_content_tool_call(
    bedrock_client,
    instrument_no_content,
    span_exporter,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "text": (
                        "What is the weather in Seattle and San Francisco"
                        " today?"
                    )
                }
            ],
        }
    ]
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_current_weather",
                    "description": (
                        "Get the current weather in a given location."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The name of the city",
                                }
                            },
                            "required": ["location"],
                        }
                    },
                }
            }
        ]
    }

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        toolConfig=tool_config,
    )

    events = list(response["stream"])
    assert len(events) > 0

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )
    assert GenAIAttributes.GEN_AI_TOOL_DEFINITIONS in (span.attributes or {})
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in (span.attributes or {})
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in (
        span.attributes or {}
    )


@pytest.mark.vcr
def test_converse_stream_close_before_consumption(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    response["stream"].close()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.status.status_code == StatusCode.UNSET


@pytest.mark.vcr
def test_converse_stream_with_invalid_model(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    with pytest.raises(ClientError) as exc_info:
        bedrock_client.converse_stream(
            messages=messages,
            modelId="does-not-exist",
        )

    assert exc_info.value.response["Error"]["Code"] == "ValidationException"

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat does-not-exist"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] in (
        "ValidationException",
        "botocore.errorfactory.ValidationException",
    )


@pytest.mark.vcr
def test_converse_stream_handles_event_stream_error(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    with mock.patch.object(
        EventStream,
        "_parse_event",
        side_effect=EventStreamError(
            {"modelStreamErrorException": {}}, "ConverseStream"
        ),
    ):
        with pytest.raises(EventStreamError):
            for _event in response["stream"]:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] in (
        "EventStreamError",
        "botocore.exceptions.EventStreamError",
    )


@pytest.mark.vcr
def test_converse_stream_caller_side_error(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse_stream(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    with pytest.raises(RuntimeError, match="caller aborted stream"):
        with response["stream"] as stream:
            for _event in stream:
                raise RuntimeError("caller aborted stream")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


def test_stream_wrapper_no_content(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    wrapper = BedrockConverseStreamWrapper(
        stream=mock.MagicMock(),
        invocation=invocation,
        capture_content=False,
    )
    wrapper._process_chunk(
        {
            "contentBlockStart": {"contentBlockIndex": 0, "start": {}},
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"text": "hello"},
            },
            "messageStop": {"stopReason": "end_turn"},
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 4,
                    "cacheWriteInputTokens": 8,
                }
            },
        }
    )
    wrapper._on_stream_end()
    assert not invocation.output_messages
    assert invocation.finish_reasons == ["stop"]
    assert invocation.response_model_name is None
    assert invocation.input_tokens == 10
    assert invocation.output_tokens == 5
    assert invocation.cache_read_input_tokens == 4
    assert invocation.cache_creation_input_tokens == 8


def test_stream_wrapper_with_reasoning(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    wrapper = BedrockConverseStreamWrapper(
        stream=mock.MagicMock(),
        invocation=invocation,
        capture_content=True,
    )
    wrapper._process_chunk(
        {
            "contentBlockStart": {
                "contentBlockIndex": 0,
                "start": {},
            },
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": "Step 1: thinking..."}},
            },
        }
    )
    wrapper._process_chunk(
        {
            "contentBlockDelta": {
                "contentBlockIndex": 0,
                "delta": {"reasoningContent": {"text": " Step 2: concluded."}},
            }
        }
    )
    wrapper._process_chunk(
        {
            "contentBlockStart": {"contentBlockIndex": 1, "start": {}},
            "contentBlockDelta": {
                "contentBlockIndex": 1,
                "delta": {"text": "Here is the result."},
            },
            "messageStop": {"stopReason": "end_turn"},
        }
    )
    wrapper._on_stream_end()
    assert len(invocation.output_messages) == 1
    out_msg = invocation.output_messages[0]
    assert len(out_msg.parts) == 2
    assert out_msg.parts[0].type == "reasoning"
    assert (
        getattr(out_msg.parts[0], "content")
        == "Step 1: thinking... Step 2: concluded."
    )
    assert out_msg.parts[1].type == "text"
    assert getattr(out_msg.parts[1], "content") == "Here is the result."
    assert invocation.finish_reasons == ["stop"]
