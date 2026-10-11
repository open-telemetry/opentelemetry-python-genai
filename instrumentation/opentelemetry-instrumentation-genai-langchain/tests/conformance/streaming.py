# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=100,
    temperature=0.5,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.2,
    stop=["\n\n"],
    seed=42,
    stream_usage=True,
)

for _ in model.stream(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Say this is a test"),
    ]
):
    pass
