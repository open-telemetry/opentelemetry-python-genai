# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock Converse API instrumentation."""

import json

import pytest
from botocore.exceptions import ClientError

from opentelemetry.instrumentation.genai.bedrock.extractors import (
    extract_content_block,
    extract_converse_request,
    extract_converse_response,
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
def test_converse_with_content(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    assert response
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

    # Check captured message content
    input_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "Say this is a test"

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert len(output_msgs[0]["parts"]) > 0


@pytest.mark.vcr
def test_converse_no_content(
    bedrock_client,
    instrument_no_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    response = bedrock_client.converse(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        inferenceConfig={
            "maxTokens": 10,
            "temperature": 0.8,
            "topP": 1,
            "stopSequences": ["|"],
        },
    )

    assert response
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "amazon.nova-micro-v1:0"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS]
        is not None
    )


@pytest.mark.vcr
def test_converse_tool_call_with_content(
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

    response = bedrock_client.converse(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        toolConfig=tool_config,
    )

    assert response
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.nova-micro-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    parts = output_msgs[0]["parts"]
    tool_calls = [p for p in parts if p.get("type") == "tool_call"]
    assert len(tool_calls) >= 1
    assert tool_calls[0]["name"] == "get_current_weather"


@pytest.mark.vcr
def test_converse_tool_call_no_content(
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

    response = bedrock_client.converse(
        messages=messages,
        modelId="amazon.nova-micro-v1:0",
        toolConfig=tool_config,
    )

    assert response
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
def test_converse_with_invalid_model(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    messages = [{"role": "user", "content": [{"text": "Say this is a test"}]}]

    with pytest.raises(ClientError) as exc_info:
        bedrock_client.converse(
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


def test_extract_converse_request_no_content(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_converse_request(
        {
            "messages": [{"role": "user", "content": [{"text": "hello"}]}],
            "system": [{"text": "system instruction"}],
            "inferenceConfig": {"temperature": 0.5},
            "toolConfig": {
                "tools": [{"toolSpec": {"name": "get_weather"}}],
            },
        },
        invocation,
        capture_content=False,
    )
    assert not invocation.input_messages
    assert not invocation.system_instruction
    assert invocation.tool_definitions
    assert invocation.temperature == 0.5


def test_extract_converse_response_no_content(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hi"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 5,
                "outputTokens": 2,
                "cacheReadInputTokens": 3,
                "cacheWriteInputTokens": 7,
            },
        },
        invocation,
        capture_content=False,
    )
    assert not invocation.output_messages
    assert invocation.finish_reasons == ["stop"]
    assert invocation.input_tokens == 5
    assert invocation.output_tokens == 2
    assert invocation.cache_read_input_tokens == 3
    assert invocation.cache_creation_input_tokens == 7


def test_extract_converse_request_top_k_and_seed(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_converse_request(
        {
            "inferenceConfig": {"topK": 40, "seed": 123},
        },
        invocation,
    )
    assert invocation.top_k == 40.0
    assert invocation.seed == 123

    invocation2 = handler.inference(provider="aws.bedrock")
    extract_converse_request(
        {
            "additionalModelRequestFields": {"top_k": 250, "seed": 456},
        },
        invocation2,
    )
    assert invocation2.top_k == 250.0
    assert invocation2.seed == 456

    invocation3 = handler.inference(provider="aws.bedrock")
    extract_converse_request(
        {
            "additionalModelRequestFields": {"inferenceConfig": {"topK": 20}},
        },
        invocation3,
    )
    assert invocation3.top_k == 20.0


def test_extract_content_block_reasoning() -> None:
    part = extract_content_block(
        {
            "reasoningContent": {
                "reasoningText": {"text": "Thinking step by step..."}
            }
        }
    )
    assert part is not None
    assert part.type == "reasoning"
    assert getattr(part, "content") == "Thinking step by step..."

    # Redacted reasoningContent
    part2 = extract_content_block(
        {"reasoningContent": {"redactedContent": b"redacted_bytes"}}
    )
    assert part2 is not None
    assert part2.type == "reasoning"
    assert getattr(part2, "content") == ""


def test_extract_content_block_document() -> None:
    doc_part = extract_content_block(
        {
            "document": {
                "format": "pdf",
                "name": "sample.pdf",
                "source": {"bytes": b"pdf_data"},
            }
        }
    )
    assert doc_part is not None
    assert doc_part.type == "blob"
    assert getattr(doc_part, "modality") == "document"
    assert getattr(doc_part, "mime_type") == "application/pdf"
    assert getattr(doc_part, "content") == b"pdf_data"


def test_extract_converse_response_reasoning(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_converse_response(
        {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "reasoningContent": {
                                "reasoningText": {"text": "Let me think"}
                            }
                        },
                        {"text": "The answer is 42."},
                    ],
                }
            },
            "stopReason": "end_turn",
        },
        invocation,
        capture_content=True,
    )
    assert len(invocation.output_messages) == 1
    out_msg = invocation.output_messages[0]
    assert len(out_msg.parts) == 2
    assert out_msg.parts[0].type == "reasoning"
    assert getattr(out_msg.parts[0], "content") == "Let me think"
    assert out_msg.parts[1].type == "text"
    assert getattr(out_msg.parts[1], "content") == "The answer is 42."


def test_extract_content_block_generic_parts() -> None:
    for key in (
        "video",
        "audio",
        "guardContent",
        "cachePoint",
        "citationsContent",
        "searchResult",
        "toolAddition",
        "toolRemoval",
    ):
        part = extract_content_block({key: {"foo": "bar"}})
        assert part is not None
        assert part.type == key
