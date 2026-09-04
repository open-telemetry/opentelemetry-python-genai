# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from _helpers import GetWeatherTool, transformers_model

tool = GetWeatherTool()
model = transformers_model()
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What is the weather in Paris?"}],
    }
]

response = model.generate(messages=messages, tools_to_call_from=[tool])
tool_result = tool(location="Paris")

messages.extend(
    [
        {
            "role": "tool-call",
            "content": [{"type": "text", "text": response.content}],
        },
        {
            "role": "tool-response",
            "content": [{"type": "text", "text": tool_result}],
        },
    ]
)

model.generate(messages=messages, tools_to_call_from=[tool])
