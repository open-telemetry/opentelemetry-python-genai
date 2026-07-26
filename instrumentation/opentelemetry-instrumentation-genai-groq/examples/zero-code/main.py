# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from groq import Groq


def main():
    client = Groq()
    chat_completion = client.chat.completions.create(
        model=os.getenv("CHAT_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {
                "role": "user",
                "content": "Write a short poem on OpenTelemetry.",
            },
        ],
    )
    print(chat_completion.choices[0].message.content)


if __name__ == "__main__":
    main()
