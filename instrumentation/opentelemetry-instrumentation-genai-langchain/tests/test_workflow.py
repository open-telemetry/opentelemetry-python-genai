# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.test_util_genai.instrumentor import instrument

langgraph_graph = pytest.importorskip("langgraph.graph")
END = langgraph_graph.END
START = langgraph_graph.START
StateGraph = langgraph_graph.StateGraph


class _State(TypedDict):
    messages: list[BaseMessage]


def _respond(_: _State) -> _State:
    return {"messages": [AIMessage(content="done")]}


def _graph(node: Any, *, name: str | None = None) -> Any:
    builder = StateGraph(_State)
    builder.add_node("step", node)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile(name=name)


def _nested_graph() -> Any:
    return _graph(_graph(_respond, name="named_subgraph"))


def _workflow_spans(span_exporter: Any) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
        == "invoke_workflow"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_mode",
    [False, True],
    ids=["sync", "async"],
)
async def test_nested_graph_emits_workflow_span(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    async_mode: bool,
) -> None:
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        graph = _nested_graph()
        inputs = {"messages": [HumanMessage(content="hello")]}
        result = (
            await graph.ainvoke(inputs) if async_mode else graph.invoke(inputs)
        )

    assert result == {"messages": [AIMessage(content="done")]}
    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2
    workflow_spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span
        for span in workflow_spans
    }
    assert set(workflow_spans_by_name) == {"LangGraph", "named_subgraph"}
    inner_span = workflow_spans_by_name["named_subgraph"]
    outer_span = workflow_spans_by_name["LangGraph"]
    assert all(
        GenAI.GEN_AI_INPUT_MESSAGES in span.attributes
        for span in workflow_spans
    )
    assert all(
        GenAI.GEN_AI_OUTPUT_MESSAGES in span.attributes
        for span in workflow_spans
    )
    if async_mode:
        pytest.xfail(
            "nested spans are not parented in async: "
            "https://github.com/open-telemetry/"
            "opentelemetry-python-genai/issues/513"
        )
    assert inner_span.parent.span_id == outer_span.context.span_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_mode",
    [False, True],
    ids=["sync", "async"],
)
@pytest.mark.parametrize(
    ("agent_metadata", "agent_span_name"),
    [
        pytest.param(
            {"agent_name": "my_agent"},
            "invoke_agent my_agent",
            id="agent-name",
        ),
        pytest.param(
            {"agent_type": "custom"},
            "invoke_agent",
            id="agent-type",
        ),
        pytest.param(
            {"otel_agent_span": True},
            "invoke_agent",
            id="agent-override",
        ),
    ],
)
async def test_nested_graph_with_bound_agent_metadata_is_agent(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    async_mode: bool,
    agent_metadata: dict[str, Any],
    agent_span_name: str,
) -> None:
    subgraph = _graph(_respond, name="my_agent").with_config(
        {
            "metadata": {
                **agent_metadata,
                "thread_id": "thread-1",
                "agent_id": "agent-1",
                "agent_description": "test agent",
            }
        }
    )
    graph = _graph(subgraph)

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="SPAN_ONLY",
    ):
        inputs = {"messages": [HumanMessage(content="hello")]}
        outer_config: dict[str, Any] = {
            "metadata": {
                "agent_name": "outer",
                "agent_id": "outer-id",
                "agent_description": "outer agent",
            }
        }
        result = (
            await graph.ainvoke(inputs, config=outer_config)
            if async_mode
            else graph.invoke(inputs, config=outer_config)
        )

    assert result == {"messages": [AIMessage(content="done")]}
    operation_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
        in {"invoke_agent", "invoke_workflow"}
    ]
    agent_spans = [
        span
        for span in operation_spans
        if span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_agent"
    ]
    workflow_spans = [
        span
        for span in operation_spans
        if span.attributes[GenAI.GEN_AI_OPERATION_NAME] == "invoke_workflow"
    ]
    assert len(agent_spans) == 1
    assert [span.name for span in workflow_spans] == [
        "invoke_workflow LangGraph"
    ]
    assert agent_spans[0].name == agent_span_name
    assert (
        agent_spans[0].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"
    )
    assert agent_spans[0].attributes[GenAI.GEN_AI_AGENT_ID] == "agent-1"
    assert (
        agent_spans[0].attributes[GenAI.GEN_AI_AGENT_DESCRIPTION]
        == "test agent"
    )
    assert GenAI.GEN_AI_INPUT_MESSAGES in agent_spans[0].attributes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "async_mode",
    [False, True],
    ids=["sync", "async"],
)
@pytest.mark.parametrize(
    ("operation_metadata", "operation_span_name"),
    [
        pytest.param(
            {"agent_name": "inner_agent"},
            "invoke_agent inner_agent",
            id="agent-name",
        ),
        pytest.param(
            {"agent_type": "custom"},
            "invoke_agent marked_node",
            id="agent-type",
        ),
        pytest.param(
            {"otel_agent_span": True},
            "invoke_agent marked_node",
            id="agent-override",
        ),
        pytest.param(
            {"otel_workflow_span": True},
            "invoke_workflow marked_node",
            id="workflow-override",
        ),
    ],
)
async def test_graph_child_with_local_operation_metadata(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    async_mode: bool,
    operation_metadata: dict[str, Any],
    operation_span_name: str,
) -> None:
    node = RunnableLambda(_respond).with_config(
        run_name="marked_node",
        metadata=operation_metadata,
    )
    graph = _graph(node)

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        inputs = {"messages": [HumanMessage(content="hello")]}
        result = (
            await graph.ainvoke(inputs) if async_mode else graph.invoke(inputs)
        )

    assert result == {"messages": [AIMessage(content="done")]}
    operation_spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME)
        in {"invoke_agent", "invoke_workflow"}
    ]
    assert {span.name for span in operation_spans} == {
        operation_span_name,
        "invoke_workflow LangGraph",
    }


def test_nested_runnable_sequence_is_not_workflow(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
) -> None:
    sequence = RunnableLambda(_respond) | RunnableLambda(_respond)

    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        _graph(sequence).invoke({"messages": [HumanMessage(content="hello")]})

    assert len(_workflow_spans(span_exporter)) == 1


@pytest.mark.parametrize(
    "agent_metadata",
    [
        pytest.param({"agent_type": "outer"}, id="agent-type"),
        pytest.param({"otel_agent_span": True}, id="agent-override"),
    ],
)
def test_nested_graph_under_inherited_agent_metadata_is_workflow(
    tracer_provider,
    meter_provider,
    logger_provider,
    span_exporter,
    agent_metadata: dict[str, Any],
) -> None:
    with instrument(
        LangChainInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    ):
        _nested_graph().invoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"metadata": agent_metadata},
        )

    spans = span_exporter.get_finished_spans()
    span_names = {span.name for span in spans}
    assert "invoke_workflow named_subgraph" in span_names
    assert "invoke_agent named_subgraph" not in span_names
    assert not [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]
    nested_span = next(
        span for span in spans if span.name == "invoke_workflow named_subgraph"
    )
    assert nested_span.parent is not None
