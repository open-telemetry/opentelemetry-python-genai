# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Bedrock Converse instrumentation."""

from __future__ import annotations

import boto3
import pytest
from botocore.stub import Stubber

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    server_attributes as ServerAttributes,
)


@pytest.fixture
def bedrock_client():
    """Create a real botocore bedrock-runtime client."""
    return boto3.client("bedrock-runtime", region_name="us-east-1")


def _converse_response():
    """Build a Converse API response for the Stubber."""
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "Hello!"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
        },
        "ResponseMetadata": {
            "RequestId": "test-request-id",
            "HTTPStatusCode": 200,
            "HTTPHeaders": {},
            "RetryAttempts": 0,
        },
        "metrics": {"latencyMs": 100},
    }


class TestConverseNoContent:
    """Test Converse instrumentation without content capture."""

    def test_basic_span_attributes(
        self, instrument_no_content, bedrock_client, span_exporter
    ):
        """Test that basic span attributes are set correctly."""
        with Stubber(bedrock_client) as stubber:
            stubber.add_response("converse", _converse_response())
            bedrock_client.converse(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=[{"role": "user", "content": [{"text": "Hello!"}]}],
            )

        span_list = span_exporter.get_finished_spans()
        assert len(span_list) == 1
        span = span_list[0]

        assert span.name == "chat anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME] == "chat"
        assert span.attributes[GenAIAttributes.GEN_AI_SYSTEM] == "aws.bedrock"
        assert (
            span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL]
            == "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 5
        assert span.attributes[
            GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS
        ] == ("stop",)
        assert ServerAttributes.SERVER_ADDRESS in span.attributes

    def test_error_records_exception(
        self, instrument_no_content, bedrock_client, span_exporter
    ):
        """Test that exceptions are recorded on the span."""
        from botocore.exceptions import (  # noqa: PLC0415
            ClientError,
        )

        with Stubber(bedrock_client) as stubber:
            stubber.add_client_error(
                "converse",
                service_error_code="ThrottlingException",
                service_message="Rate exceeded",
            )
            with pytest.raises(ClientError):
                bedrock_client.converse(
                    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                    messages=[{"role": "user", "content": [{"text": "Hi"}]}],
                )

        span_list = span_exporter.get_finished_spans()
        assert len(span_list) == 1
        span = span_list[0]
        assert span.status.is_ok is False


class TestConverseWithContent:
    """Test Converse instrumentation with content capture."""

    def test_captures_output_text(
        self, instrument_with_content, bedrock_client, span_exporter
    ):
        """Test that output message content is captured."""
        with Stubber(bedrock_client) as stubber:
            stubber.add_response("converse", _converse_response())
            bedrock_client.converse(
                modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
                messages=[{"role": "user", "content": [{"text": "Say hi"}]}],
            )

        span_list = span_exporter.get_finished_spans()
        assert len(span_list) == 1
