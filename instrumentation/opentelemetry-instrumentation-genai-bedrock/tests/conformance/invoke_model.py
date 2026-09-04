# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os

import boto3

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    endpoint_url=os.environ.get("AWS_ENDPOINT_URL_BEDROCK_RUNTIME"),
)

response = client.invoke_model(
    modelId="amazon.titan-text-express-v1",
    body=json.dumps(
        {
            "inputText": "Say this is a test",
            "textGenerationConfig": {
                "maxTokenCount": 10,
                "temperature": 0.8,
                "topP": 1.0,
                "stopSequences": ["|"],
            },
        }
    ),
)
response["body"].read()
