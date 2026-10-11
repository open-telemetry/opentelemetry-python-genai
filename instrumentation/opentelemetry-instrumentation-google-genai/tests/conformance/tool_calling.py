# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Two model turns with caller-managed tool execution."""

from google.genai import Client, types

client = Client()
config = types.GenerateContentConfig(
    tools=[
        types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="get_weather",
                    description="Get weather for a location",
                    parameters={
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                )
            ]
        )
    ],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
)
contents = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the weather in Boston?")],
    )
]
response = client.models.generate_content(
    model="gemini-2.5-flash", contents=contents, config=config
)
assert response.function_calls
contents.append(response.candidates[0].content)
contents.append(
    types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=call.name, response={"result": "sunny"}
            )
            for call in response.function_calls
        ],
    )
)
client.models.generate_content(
    model="gemini-2.5-flash", contents=contents, config=config
)
