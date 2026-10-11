# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock retrieve."""

from __future__ import annotations

import boto3

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
client.retrieve(
    knowledgeBaseId="KB1234567890",
    retrievalQuery={"text": "What is the capital of France?"},
    retrievalConfiguration={
        "vectorSearchConfiguration": {"numberOfResults": 2}
    },
)
