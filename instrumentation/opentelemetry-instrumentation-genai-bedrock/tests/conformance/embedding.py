# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: bedrock invoke_model (embeddings)."""

from __future__ import annotations

import json

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
response = client.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body=json.dumps(
        {"inputText": "The quick brown fox jumps over the lazy dog"}
    ),
)
response["body"].read()
