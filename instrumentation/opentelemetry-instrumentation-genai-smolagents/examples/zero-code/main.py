# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from smolagents import CodeAgent, InferenceClientModel


def main():
    agent = CodeAgent(
        tools=[],
        model=InferenceClientModel(
            model_id=os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
        ),
    )
    print(
        agent.run(
            os.getenv("SMOLAGENTS_TASK", "How many seconds are in a week?")
        )
    )


if __name__ == "__main__":
    main()
