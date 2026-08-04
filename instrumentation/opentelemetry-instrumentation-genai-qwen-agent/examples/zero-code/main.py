# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import os

from qwen_agent.agents import Assistant


def main():
    bot = Assistant(
        llm={
            "model": os.getenv("CHAT_MODEL", "qwen-max"),
            "model_type": "qwen_dashscope",
        },
        name="example-assistant",
        system_message="You are a helpful assistant.",
    )
    responses = []
    for responses in bot.run(
        [{"role": "user", "content": "Write a short poem on OpenTelemetry."}]
    ):
        pass
    for message in responses:
        print(message)


if __name__ == "__main__":
    main()
