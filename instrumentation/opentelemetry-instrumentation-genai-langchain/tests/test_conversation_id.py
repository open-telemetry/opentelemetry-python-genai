# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conversation IDs across real LangChain run trees, without provider calls."""

from __future__ import annotations

from typing import Any

import langchain.agents
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from tests._api import FakeChatModel, FakeRetriever, finished_chat, invoke

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import StatusCode

_CREATE_AGENT = getattr(langchain.agents, "create_agent", None)


def _spans_by_operation(
    exporter: InMemorySpanExporter,
) -> dict[str, ReadableSpan]:
    spans = exporter.get_finished_spans()
    assert all(s.status.status_code == StatusCode.UNSET for s in spans)
    assert all(s.end_time is not None for s in spans)
    result = {s.attributes[GenAI.GEN_AI_OPERATION_NAME]: s for s in spans}
    assert len(result) == len(spans)
    finished_chat(exporter)
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("agent", [False, True], ids=["workflow", "agent"])
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {"thread_id": "t1", "session_id": "s1", "conversation_id": "c1"},
            "t1",
        ),
        ({"session_id": "s1", "conversation_id": "c1"}, "s1"),
        ({"conversation_id": "c1"}, "c1"),
        ({}, None),
    ],
)
async def test_conversation_id_metadata_and_precedence(
    asynchronous: bool,
    agent: bool,
    metadata: dict[str, str],
    expected: str | None,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(responses=[AIMessage(content="12")])

    def run(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {"messages": [model.invoke("What is 3 * 4?", config=config)]}

    async def arun(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {
            "messages": [await model.ainvoke("What is 3 * 4?", config=config)]
        }

    root = RunnableLambda(run, afunc=arun).with_config(
        run_name="math_agent" if agent else "MyLangGraph",
        metadata=metadata,
    )
    assert (await invoke(root, {}, asynchronous))["messages"][
        0
    ].content == "12"
    spans = _spans_by_operation(span_exporter)
    operation = "invoke_agent" if agent else "invoke_workflow"
    assert set(spans) == {operation, "chat"}
    for span in spans.values():
        assert span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID) == expected
    assert spans["chat"].parent is not None
    assert spans["chat"].parent.span_id == spans[operation].context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("agent", [False, True], ids=["workflow", "agent"])
@pytest.mark.parametrize(
    "child_id", [None, "inner"], ids=["context-inheritance", "child-override"]
)
async def test_child_conversation_id_resolution(
    asynchronous: bool,
    agent: bool,
    child_id: str | None,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(responses=[AIMessage(content="12")])

    def child_config(config: RunnableConfig) -> RunnableConfig:
        # Invocation config overrides LangChain's inherited metadata; a null
        # value leaves conversation resolution to the invocation's OTel context.
        return {
            **config,
            "metadata": {**config.get("metadata", {}), "thread_id": child_id},
        }

    def run(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {
            "messages": [
                model.invoke("Calculate", config=child_config(config))
            ]
        }

    async def arun(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {
            "messages": [
                await model.ainvoke("Calculate", config=child_config(config))
            ]
        }

    root = RunnableLambda(run, afunc=arun).with_config(
        run_name="math_agent" if agent else "MyLangGraph",
        metadata={"thread_id": "outer"},
    )
    assert (await invoke(root, {}, asynchronous))["messages"][
        0
    ].content == "12"
    spans = _spans_by_operation(span_exporter)
    root_span = spans["invoke_agent" if agent else "invoke_workflow"]
    assert root_span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "outer"
    assert spans["chat"].attributes[GenAI.GEN_AI_CONVERSATION_ID] == (
        child_id or "outer"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("layer", ["tool", "retrieval", "internal-chain"])
async def test_intermediate_operations_do_not_lose_conversation_id(
    asynchronous: bool,
    layer: str,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    model = FakeChatModel(responses=[AIMessage(content="12")])

    @tool
    def calculate(question: str) -> str:
        """Calculate using a local model."""
        return str(model.invoke(question).content)

    retriever = FakeRetriever(
        documents=[Document(page_content="12")], model=model
    )

    def chat(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {"messages": [model.invoke("Calculate", config=config)]}

    async def achat(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        return {"messages": [await model.ainvoke("Calculate", config=config)]}

    internal = RunnableLambda(chat, afunc=achat).with_config(
        run_name="internal"
    )

    def run(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        if layer == "tool":
            result = calculate.invoke({"question": "Calculate"}, config=config)
        elif layer == "retrieval":
            result = retriever.invoke("Calculate", config=config)
        else:
            return internal.invoke(state, config=config)
        return {"result": result}

    async def arun(
        state: dict[str, object], config: RunnableConfig
    ) -> dict[str, object]:
        if layer == "tool":
            result = await calculate.ainvoke(
                {"question": "Calculate"}, config=config
            )
        elif layer == "retrieval":
            result = await retriever.ainvoke("Calculate", config=config)
        else:
            return await internal.ainvoke(state, config=config)
        return {"result": result}

    root = RunnableLambda(run, afunc=arun).with_config(
        run_name="math_agent",
        metadata={"agent_name": "math_agent", "thread_id": "t1"},
    )
    result = await invoke(root, {}, asynchronous)
    assert result
    spans = _spans_by_operation(span_exporter)
    assert (
        spans["invoke_agent"].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "t1"
    )
    assert spans["chat"].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "t1"
    if layer == "internal-chain":
        assert set(spans) == {"invoke_agent", "chat"}
        parent = spans["invoke_agent"]
    else:
        operation = "execute_tool" if layer == "tool" else "retrieval"
        assert set(spans) == {"invoke_agent", "chat", operation}
        parent = spans[operation]
        assert GenAI.GEN_AI_CONVERSATION_ID not in parent.attributes
        assert parent.parent is not None
        assert parent.parent.span_id == spans["invoke_agent"].context.span_id
    assert spans["chat"].parent is not None
    assert spans["chat"].parent.span_id == parent.context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.skipif(
    _CREATE_AGENT is None, reason="create_agent requires LangChain 1.x"
)
async def test_unnamed_agent_conversation_id(
    asynchronous: bool,
    span_exporter: InMemorySpanExporter,
    instrument_langchain: None,
) -> None:
    assert _CREATE_AGENT is not None
    agent = _CREATE_AGENT(
        FakeChatModel(responses=[AIMessage(content="12")]), tools=[]
    )
    result: Any = await invoke(
        agent,
        {"messages": [HumanMessage(content="Calculate")]},
        asynchronous,
        config={"metadata": {"thread_id": "t1"}},
    )
    assert result["messages"][-1].content == "12"
    spans = _spans_by_operation(span_exporter)
    assert set(spans) == {"invoke_agent", "chat"}
    assert spans["invoke_agent"].name == "invoke_agent"
    assert GenAI.GEN_AI_AGENT_NAME not in spans["invoke_agent"].attributes
    assert (
        spans["invoke_agent"].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "t1"
    )
    assert spans["chat"].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "t1"
