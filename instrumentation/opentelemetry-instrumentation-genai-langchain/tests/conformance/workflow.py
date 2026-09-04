# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ]
)
chain = (
    prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_tokens=100)
).with_config(run_name="conformance_workflow")

chain.invoke({"question": "Say this is a test"})
