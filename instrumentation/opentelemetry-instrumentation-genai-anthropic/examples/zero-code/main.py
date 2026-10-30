# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import anthropic


def main():
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Write a short poem on OpenTelemetry."}
        ],
    )
    print(next(b.text for b in message.content if b.type == "text"))


if __name__ == "__main__":
    main()
