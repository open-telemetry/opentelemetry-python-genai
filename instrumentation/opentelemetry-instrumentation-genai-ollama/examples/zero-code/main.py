# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

import ollama


def main():
    response = ollama.chat(
        model=os.getenv("CHAT_MODEL", "qwen2.5:0.5b"),
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
    print(response.message.content)


if __name__ == "__main__":
    main()
