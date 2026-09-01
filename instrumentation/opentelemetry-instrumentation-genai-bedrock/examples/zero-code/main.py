# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file

import boto3


def main():
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=[
            {
                "role": "user",
                "content": [{"text": "Write a short poem on OpenTelemetry."}],
            }
        ],
    )
    print(response)


if __name__ == "__main__":
    main()
