# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock retrieve_and_generate_stream."""

from __future__ import annotations

import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
response = client.retrieve_and_generate_stream(
    input={"text": "What is the capital of France?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "KB1234567890",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0",
        },
    },
    sessionId="conformance-session-1",
)
for _ in response["stream"]:
    pass
