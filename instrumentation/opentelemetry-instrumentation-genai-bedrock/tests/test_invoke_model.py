# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock InvokeModel API instrumentation."""

from __future__ import annotations

import io
import json

import pytest
from botocore.exceptions import ClientError
from botocore.response import StreamingBody
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock.extractors import (
    extract_invoke_model_request,
    extract_invoke_model_response,
    is_embedding_model,
)
from opentelemetry.semconv._incubating.attributes import (
    aws_attributes as AwsAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.metrics import (
    gen_ai_metrics as GenAIMetrics,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


def test_invoke_model_anthropic_messages(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "Hello!"}],
    }
    response_body = {
        "id": "msg_12345",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hi there!"}],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "cache_creation_input_tokens": 2,
            "cache_read_input_tokens": 4,
        },
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(request_body),
        )

    # Verify StreamingBody is preserved and readable
    assert response["body"].read() == raw_response_bytes

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
    assert GenAIAttributes.GEN_AI_RESPONSE_MODEL not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "msg_12345"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 100
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.7
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.9
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 40
    assert isinstance(
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_K], int
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 12
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert span.attributes["gen_ai.usage.cache_write.input_tokens"] == 2
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 4
    )
    assert (
        span.attributes[ServerAttributes.SERVER_ADDRESS]
        == "bedrock-runtime.us-east-1.amazonaws.com"
    )

    # Verify captured content
    input_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "Hello!"

    sys_instruction = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS]
    )
    assert sys_instruction[0]["content"] == "You are a helpful assistant."

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "Hi there!"


def test_invoke_model_anthropic_legacy_completion(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "\n\nHuman: Tell me a joke\n\nAssistant:",
        "max_tokens_to_sample": 50,
        "temperature": 0.5,
        "stop_sequences": ["\n\nHuman:"],
    }
    response_body = {
        "completion": " Why did the chicken cross the road?",
        "stop_reason": "stop_sequence",
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-v2",
            "body": json.dumps(request_body).encode("utf-8"),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(request_body).encode("utf-8"),
        )

    assert response["body"].read() == raw_response_bytes
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat anthropic.claude-v2"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 50
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_STOP_SEQUENCES] == (
        "\n\nHuman:",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_titan_text(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "inputText": "Write a poem",
        "textGenerationConfig": {
            "maxTokenCount": 200,
            "temperature": 0.6,
            "topP": 0.8,
            "stopSequences": ["User:"],
        },
    }
    response_body = {
        "inputTextTokenCount": 3,
        "results": [
            {
                "tokenCount": 25,
                "outputText": "Roses are red, violets are blue...",
                "completionReason": "FINISH",
            }
        ],
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "amazon.titan-text-express-v1",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-text-express-v1",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat amazon.titan-text-express-v1"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 200
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.6
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TOP_P] == 0.8
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 25
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_llama(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "Explain quantum computing",
        "max_gen_len": 150,
        "temperature": 0.5,
        "top_p": 0.9,
    }
    response_body = {
        "generation": "Quantum computing uses qubits...",
        "prompt_token_count": 5,
        "generation_token_count": 30,
        "stop_reason": "stop",
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "meta.llama3-8b-instruct-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="meta.llama3-8b-instruct-v1:0",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat meta.llama3-8b-instruct-v1:0"
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 150
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 30
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )


def test_invoke_model_no_content(
    bedrock_client,
    instrument_no_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "messages": [{"role": "user", "content": "Sensitive secret"}],
        "max_tokens": 50,
    }
    response_body = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Sensitive answer"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 10},
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS not in span.attributes
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10


def test_invoke_model_error(
    bedrock_client,
    instrument_bedrock,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    stubber.add_client_error(
        "invoke_model",
        service_error_code="ValidationException",
        service_message="Model identifier is invalid",
        expected_params={
            "modelId": "invalid-model-id",
            "body": b'{"prompt": "hi"}',
        },
    )

    with stubber:
        with pytest.raises(ClientError):
            bedrock_client.invoke_model(
                modelId="invalid-model-id",
                body=b'{"prompt": "hi"}',
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] in (
        "ValidationException",
        "botocore.errorfactory.ValidationException",
    )


def test_extract_invoke_model_response_headers(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_invoke_model_response(
        {
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "X-Amzn-Bedrock-Input-Token-Count": "15",
                    "X-Amzn-Bedrock-Output-Token-Count": "22",
                }
            }
        },
        b'{"completion": "hello"}',
        invocation,
    )
    assert invocation.input_tokens == 15
    assert invocation.output_tokens == 22


def test_extract_invoke_model_request_zero_values(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")
    extract_invoke_model_request(
        {
            "body": json.dumps(
                {
                    "temperature": 0.0,
                    "top_p": 0.0,
                    "top_k": 0,
                    "max_tokens": 0,
                    "seed": 0,
                }
            )
        },
        invocation,
    )
    assert invocation.temperature == 0.0
    assert invocation.top_p == 0.0
    assert invocation.top_k == 0
    assert isinstance(invocation.top_k, int)
    assert invocation.max_tokens == 0
    assert invocation.seed == 0


def test_invoke_model_anthropic_tool_call_and_result(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_prev",
                        "content": "72 degrees and sunny",
                    }
                ],
            }
        ],
    }
    response_body = {
        "id": "msg_tool_resp",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_next",
                "name": "get_weather",
                "input": {"city": "Seattle"},
            }
        ],
        "model": "claude-3-sonnet-20240229",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 25,
            "output_tokens": 15,
        },
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(request_body),
        )

    assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "tool_call",
    )

    input_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_INPUT_MESSAGES]
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["type"] == "tool_call_response"
    assert input_msgs[0]["parts"][0]["id"] == "toolu_prev"
    assert input_msgs[0]["parts"][0]["response"] == "72 degrees and sunny"

    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["finish_reason"] == "tool_call"
    assert output_msgs[0]["parts"][0]["type"] == "tool_call"
    assert output_msgs[0]["parts"][0]["id"] == "toolu_next"
    assert output_msgs[0]["parts"][0]["name"] == "get_weather"
    assert output_msgs[0]["parts"][0]["arguments"] == {"city": "Seattle"}


def test_extract_invoke_model_request_guardrail(tracer_provider) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")

    extract_invoke_model_request(
        {
            "guardrailIdentifier": "sgi5gkybzqak",
            "body": json.dumps({"prompt": "Hello"}),
        },
        invocation,
    )

    assert (
        invocation.attributes.get(AwsAttributes.AWS_BEDROCK_GUARDRAIL_ID)
        == "sgi5gkybzqak"
    )


def test_invoke_model_with_guardrail_stubber(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {"prompt": "Hello"}
    response_body = {"completion": "Hi!"}
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "anthropic.claude-v2",
            "body": json.dumps(request_body),
            "guardrailIdentifier": "sgi5gkybzqak",
        },
    )

    with stubber:
        bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(request_body),
            guardrailIdentifier="sgi5gkybzqak",
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert (
        spans[0].attributes.get(AwsAttributes.AWS_BEDROCK_GUARDRAIL_ID)
        == "sgi5gkybzqak"
    )


def test_invoke_model_mistral(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "<s>[INST] What is the capital of France? [/INST]",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response_body = {
        "outputs": [
            {
                "text": "The capital of France is Paris.",
                "stop_reason": "stop",
            }
        ]
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "15",
                    "x-amzn-bedrock-output-token-count": "7",
                }
            },
        },
        expected_params={
            "modelId": "mistral.mistral-7b-instruct-v0:2",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="mistral.mistral-7b-instruct-v0:2",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat mistral.mistral-7b-instruct-v0:2"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 15
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 7
    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert (
        output_msgs[0]["parts"][0]["content"]
        == "The capital of France is Paris."
    )


def test_invoke_model_cohere(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "prompt": "Write a haiku",
        "max_tokens": 50,
        "temperature": 0.3,
        "p": 0.75,
        "k": 10,
        "stop_sequences": ["--"],
    }
    response_body = {
        "id": "cohere-req-123",
        "generations": [
            {
                "id": "gen-1",
                "text": "Spring brings blossoms pink",
                "finish_reason": "COMPLETE",
            }
        ],
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "5",
                    "x-amzn-bedrock-output-token-count": "6",
                }
            },
        },
        expected_params={
            "modelId": "cohere.command-text-v14",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="cohere.command-text-v14",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "chat cohere.command-text-v14"
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 6
    output_msgs = json.loads(
        span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
    )
    assert (
        output_msgs[0]["parts"][0]["content"] == "Spring brings blossoms pink"
    )


def test_is_embedding_model() -> None:
    assert is_embedding_model("amazon.titan-embed-text-v1") is True
    assert is_embedding_model("amazon.titan-embed-text-v2:0") is True
    assert is_embedding_model("amazon.titan-embed-image-v1") is True
    assert is_embedding_model("cohere.embed-english-v3") is True
    assert is_embedding_model("cohere.embed-multilingual-v3") is True
    assert is_embedding_model("twelvelabs.marengo-embed-2-6") is True
    assert (
        is_embedding_model(
            "arn:aws:bedrock:us-east-1:123456789012:foundation-model/amazon.titan-embed-text-v1"
        )
        is True
    )
    assert is_embedding_model("us.amazon.titan-embed-text-v1") is True
    assert is_embedding_model("AMAZON.TITAN-EMBED-TEXT-V1") is True
    assert (
        is_embedding_model("anthropic.claude-3-haiku-20240307-v1:0") is False
    )
    assert is_embedding_model("amazon.titan-text-express-v1") is False
    assert is_embedding_model("meta.llama3-8b-instruct-v1:0") is False
    assert is_embedding_model("") is False
    assert is_embedding_model(None) is False


def test_invoke_model_titan_embeddings(
    bedrock_client,
    instrument_with_content,
    span_exporter,
    metric_reader,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "inputText": "This is the text to embed.",
    }
    response_body = {
        "embedding": [0.1, 0.2, 0.3, 0.4],
        "inputTextTokenCount": 7,
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "7",
                }
            },
        },
        expected_params={
            "modelId": "amazon.titan-embed-text-v1",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
        == "amazon.titan-embed-text-v1"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 4
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 7

    # Verify metrics
    metrics = metric_reader.get_metrics_data().resource_metrics
    assert len(metrics) == 1
    metric_data = metrics[0].scope_metrics[0].metrics
    duration_metric = next(
        (
            m
            for m in metric_data
            if m.name == GenAIMetrics.GEN_AI_CLIENT_OPERATION_DURATION
        ),
        None,
    )
    assert duration_metric is not None
    token_metric = next(
        (
            m
            for m in metric_data
            if m.name == GenAIMetrics.GEN_AI_CLIENT_TOKEN_USAGE
        ),
        None,
    )
    assert token_metric is not None
    assert any(
        p.attributes.get(GenAIAttributes.GEN_AI_TOKEN_TYPE)
        == GenAIAttributes.GenAiTokenTypeValues.INPUT.value
        and p.sum == 7
        for p in token_metric.data.data_points
    )


def test_invoke_model_titan_v2_embeddings_with_dimensions(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "inputText": "Hello world",
        "dimensions": 512,
        "normalize": True,
    }
    response_body = {
        "embedding": [0.0] * 512,
        "inputTextTokenCount": 3,
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "amazon.titan-embed-text-v2:0",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "embeddings amazon.titan-embed-text-v2:0"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT]
        == 512
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3


def test_invoke_model_cohere_embeddings(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "texts": ["Hello", "World"],
        "input_type": "search_document",
        "embedding_types": ["float"],
    }
    response_body = {
        "id": "emb_12345",
        "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
        "meta": {
            "billed_units": {
                "input_tokens": 10,
            }
        },
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "cohere.embed-english-v3",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="cohere.embed-english-v3",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.name == "embeddings cohere.embed-english-v3"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert span.attributes[
        GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS
    ] == ("float",)
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 2
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10


@pytest.mark.parametrize(
    "format_field,format_value",
    [
        ("embedding_types", []),
        ("embedding_types", [None]),
        ("encoding_format", ""),
    ],
)
def test_invoke_model_embeddings_omits_empty_encoding_formats(
    bedrock_client,
    instrument_with_content,
    span_exporter,
    format_field,
    format_value,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {
        "texts": ["Hello"],
        "input_type": "search_document",
        format_field: format_value,
    }
    response_body = {
        "id": "emb_12345",
        "embeddings": {"float": [[0.1, 0.2]]},
        "meta": {"billed_units": {"input_tokens": 10}},
    }
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
        },
        expected_params={
            "modelId": "cohere.embed-english-v3",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="cohere.embed-english-v3",
            body=json.dumps(request_body),
        )
        assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        GenAIAttributes.GEN_AI_REQUEST_ENCODING_FORMATS not in span.attributes
    )
    # The rest of the embedding attributes are still recorded.
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 2
    )


def test_invoke_model_embedding_error(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {"inputText": "Hello"}

    stubber.add_client_error(
        "invoke_model",
        service_error_code="ValidationException",
        service_message="Invalid input",
        expected_params={
            "modelId": "amazon.titan-embed-text-v1",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        with pytest.raises(ClientError):
            bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=json.dumps(request_body),
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert span.status.status_code == StatusCode.ERROR
    assert (
        span.attributes[ErrorAttributes.ERROR_TYPE]
        == "botocore.errorfactory.ValidationException"
    )


def test_invoke_model_embedding_body_read_error(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    class _FailingRawStream(io.RawIOBase):
        def read(self, size: int = -1) -> bytes:
            raise ConnectionError("stream read failed")

    stubber = Stubber(bedrock_client)
    request_body = json.dumps({"inputText": "Hello"})
    stubber.add_response(
        "invoke_model",
        {
            "contentType": "application/json",
            "body": StreamingBody(
                raw_stream=_FailingRawStream(),
                content_length=10,
            ),
        },
        {
            "modelId": "amazon.titan-embed-text-v1",
            "body": request_body,
        },
    )

    with stubber:
        with pytest.raises(ConnectionError, match="stream read failed"):
            bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=request_body,
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "embeddings amazon.titan-embed-text-v1"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ConnectionError"


def test_extract_invoke_model_response_prompt_cache_headers(
    tracer_provider,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")

    response = {
        "ResponseMetadata": {
            "HTTPHeaders": {
                "x-amzn-bedrock-input-token-count": "25",
                "x-amzn-bedrock-output-token-count": "15",
                "x-amzn-bedrock-cache-read-input-token-count": "10",
                "x-amzn-bedrock-cache-write-input-token-count": "5",
            }
        }
    }
    raw_body = json.dumps({"completion": "Hello!"}).encode("utf-8")

    extract_invoke_model_response(
        response,
        raw_body,
        invocation,
    )

    assert invocation.input_tokens == 25
    assert invocation.output_tokens == 15
    assert invocation.cache_read_input_tokens == 10
    assert invocation.cache_creation_input_tokens == 5


def test_extract_invoke_model_response_invocation_metrics(
    tracer_provider,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")

    raw_body = json.dumps(
        {
            "completion": "Hello!",
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 30,
                "outputTokenCount": 20,
                "cacheReadInputTokenCount": 12,
                "cacheWriteInputTokenCount": 6,
            },
        }
    ).encode("utf-8")

    extract_invoke_model_response(
        {},
        raw_body,
        invocation,
    )

    assert invocation.input_tokens == 30
    assert invocation.output_tokens == 20
    assert invocation.cache_read_input_tokens == 12
    assert invocation.cache_creation_input_tokens == 6


def test_extract_invoke_model_response_metadata_usage(
    tracer_provider,
) -> None:
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.inference(provider="aws.bedrock")

    raw_body = json.dumps(
        {
            "completion": "Hello!",
            "metadata": {
                "usage": {
                    "inputTokens": 40,
                    "outputTokens": 18,
                    "cacheReadInputTokenCount": 15,
                    "cacheWriteInputTokenCount": 8,
                }
            },
        }
    ).encode("utf-8")

    extract_invoke_model_response(
        {},
        raw_body,
        invocation,
    )

    assert invocation.input_tokens == 40
    assert invocation.output_tokens == 18
    assert invocation.cache_read_input_tokens == 15
    assert invocation.cache_creation_input_tokens == 8


def test_invoke_model_prompt_cache_headers_recorded_on_span(
    bedrock_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(bedrock_client)
    request_body = {"prompt": "Hello"}
    response_body = {"completion": "Hi there!"}
    raw_response_bytes = json.dumps(response_body).encode("utf-8")

    stubber.add_response(
        "invoke_model",
        service_response={
            "contentType": "application/json",
            "body": StreamingBody(
                io.BytesIO(raw_response_bytes), len(raw_response_bytes)
            ),
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "20",
                    "x-amzn-bedrock-output-token-count": "10",
                    "x-amzn-bedrock-cache-read-input-token-count": "14",
                    "x-amzn-bedrock-cache-write-input-token-count": "6",
                }
            },
        },
        expected_params={
            "modelId": "anthropic.claude-v2",
            "body": json.dumps(request_body),
        },
    )

    with stubber:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-v2",
            body=json.dumps(request_body),
        )

    assert response["body"].read() == raw_response_bytes

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 20
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 10
    assert (
        span.attributes[GenAIAttributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS]
        == 14
    )
    assert span.attributes["gen_ai.usage.cache_write.input_tokens"] == 6
