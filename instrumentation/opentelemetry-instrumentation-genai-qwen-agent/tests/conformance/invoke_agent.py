# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from qwen_agent.agents import Assistant

agent = Assistant(
    llm={
        "model": "gpt-4o-mini",
        "model_type": "oai",
        "generate_cfg": {"temperature": 0.5, "max_tokens": 100},
    },
    name="weather_assistant",
    system_message="You are a helpful assistant.",
)

list(agent.run([{"role": "user", "content": "Say this is a test"}]))
