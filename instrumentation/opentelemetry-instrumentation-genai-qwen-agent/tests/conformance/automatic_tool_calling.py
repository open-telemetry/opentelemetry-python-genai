# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json

from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool


@register_tool("get_current_weather")
class GetCurrentWeather(BaseTool):
    description = "Get the current weather in a given location."
    parameters = [
        {
            "name": "location",
            "type": "string",
            "description": "The city to get the weather for.",
            "required": True,
        }
    ]

    def call(self, params, **kwargs):
        location = json.loads(params)["location"]
        return f"70 degrees and sunny in {location}"


agent = Assistant(
    llm={
        "model": "gpt-4o-mini",
        "model_type": "oai",
        "generate_cfg": {"temperature": 0.5, "max_tokens": 100},
    },
    name="weather_assistant",
    system_message="You are a helpful assistant.",
    function_list=["get_current_weather"],
)

list(
    agent.run(
        [{"role": "user", "content": "What's the weather in Seattle today?"}]
    )
)
