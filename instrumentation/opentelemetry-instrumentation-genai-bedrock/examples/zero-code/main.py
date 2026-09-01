# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import json

import boto3


def main():
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.invoke_model(
        modelId="amazon.titan-text-lite-v1",
        body=json.dumps(
            {
                "inputText": "Write a short poem on OpenTelemetry.",
            }
        ),
    )
    result = json.loads(response["body"].read())
    print(result)


if __name__ == "__main__":
    main()
