# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
client.converse(
    modelId="amazon.nova-micro-v1:0",
    system=[{"text": "You are a helpful assistant."}],
    messages=[
        {
            "role": "user",
            "content": [{"text": "Say this is a test"}],
        }
    ],
    inferenceConfig={
        "maxTokens": 10,
        "temperature": 0.8,
        "topP": 1,
        "stopSequences": ["|"],
    },
)
