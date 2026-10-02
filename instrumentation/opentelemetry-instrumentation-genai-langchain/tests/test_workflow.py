# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind
from opentelemetry.trace.status import StatusCode

langgraph_graph = pytest.importorskip("langgraph.graph")
END = langgraph_graph.END
START = langgraph_graph.START
StateGraph = langgraph_graph.StateGraph


class _State(TypedDict):
    messages: list[BaseMessage]


def _respond(_: _State) -> _State:
    return {"messages": [AIMessage(content="done")]}


def _fail_step(_: _State) -> _State:
    raise RuntimeError("simulated node failure")


def _graph(node: Any, *, name: str | None = None) -> Any:
    builder = StateGraph(_State)
    builder.add_node("step", node)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    return builder.compile(name=name)


def _nested_graph() -> Any:
    return _graph(_graph(_respond, name="named_subgraph"))


@pytest.fixture
def instrument_langchain(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
):
    def _instrument(*, content_capture: str | None = None):
        return instrument(
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture=content_capture,
        )

    return _instrument


def _workflow_spans(span_exporter: InMemorySpanExporter) -> list[Any]:
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


def _agent_spans(span_exporter: InMemorySpanExporter) -> list[Any]:
    return [
        span
        for span in span_exporter.get_finished_spans()
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    ]


# ---------------------------------------------------------------------------
# 1. Top-level workflow baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("compiled_name", "expected_workflow_name"),
    [
        (None, "LangGraph"),
        ("custom_pipeline", "custom_pipeline"),
    ],
    ids=["default-name", "custom-name"],
)
def test_top_level_workflow(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
    compiled_name: str | None,
    expected_workflow_name: str,
) -> None:
    graph = _graph(_respond, name=compiled_name)

    with instrument_langchain():
        graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].name == f"invoke_workflow {expected_workflow_name}"
    assert (
        spans[0].attributes[GenAI.GEN_AI_WORKFLOW_NAME]
        == expected_workflow_name
    )
    assert spans[0].kind == SpanKind.INTERNAL


# ---------------------------------------------------------------------------
# 2. Nested subgraphs (2-level: outer -> inner)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stream_mode", [False, True], ids=["invoke", "stream"]
)
def test_nested_subgraph_emits_workflow_spans(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
    stream_mode: bool,
) -> None:
    subgraph = _graph(_respond, name="named_subgraph")
    graph = _graph(subgraph, name="outer_graph")

    with instrument_langchain(content_capture="SPAN_ONLY"):
        inputs = {"messages": [HumanMessage(content="hello")]}
        if stream_mode:
            _ = list(graph.stream(inputs))
        else:
            graph.invoke(inputs)

    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 2

    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span
        for span in workflow_spans
    }
    assert set(spans_by_name) == {"outer_graph", "named_subgraph"}

    outer_span = spans_by_name["outer_graph"]
    inner_span = spans_by_name["named_subgraph"]

    assert all(
        GenAI.GEN_AI_INPUT_MESSAGES in span.attributes
        for span in workflow_spans
    )
    assert all(
        GenAI.GEN_AI_OUTPUT_MESSAGES in span.attributes
        for span in workflow_spans
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
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
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
    assert len(agent_spans) == 2
    assert not workflow_spans
    assert {span.name for span in agent_spans} == {
        agent_span_name,
        "invoke_agent outer",
    }
    inner_span = next(
        span for span in agent_spans if span.name == agent_span_name
    )
    assert inner_span.kind == SpanKind.INTERNAL
    assert inner_span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"
    assert GenAI.GEN_AI_AGENT_ID not in inner_span.attributes
    assert (
        inner_span.attributes[GenAI.GEN_AI_AGENT_DESCRIPTION] == "test agent"
    )
    assert GenAI.GEN_AI_INPUT_MESSAGES in inner_span.attributes


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
    assert [
        span.name
        for span in spans
        if span.attributes
        and span.attributes.get(GenAI.GEN_AI_OPERATION_NAME) == "invoke_agent"
    ] == ["invoke_agent"]
    nested_span = next(
        span for span in spans if span.name == "invoke_workflow named_subgraph"
    )
    assert nested_span.parent is not None


# ---------------------------------------------------------------------------
# 3. Deeply nested subgraphs (3-level: outer -> middle -> inner)
# ---------------------------------------------------------------------------


def test_deeply_nested_subgraphs(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    inner = _graph(_respond, name="inner_workflow")
    middle = _graph(inner, name="middle_workflow")
    outer = _graph(middle, name="outer_workflow")

    with instrument_langchain():
        outer.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 3

    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span for span in spans
    }
    assert set(spans_by_name) == {
        "outer_workflow",
        "middle_workflow",
        "inner_workflow",
    }

    outer_span = spans_by_name["outer_workflow"]
    middle_span = spans_by_name["middle_workflow"]
    inner_span = spans_by_name["inner_workflow"]

    assert middle_span.parent.span_id == outer_span.context.span_id
    assert inner_span.parent.span_id == middle_span.context.span_id


# ---------------------------------------------------------------------------
# 4. Sibling subgraphs in a single workflow
# ---------------------------------------------------------------------------


def test_sibling_subgraphs_in_workflow(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    subgraph_a = _graph(_respond, name="subgraph_a")
    subgraph_b = _graph(_respond, name="subgraph_b")

    builder = StateGraph(_State)
    builder.add_node("branch_a", subgraph_a)
    builder.add_node("branch_b", subgraph_b)
    builder.add_edge(START, "branch_a")
    builder.add_edge("branch_a", "branch_b")
    builder.add_edge("branch_b", END)
    graph = builder.compile(name="parent_workflow")

    with instrument_langchain():
        graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 3

    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span for span in spans
    }
    assert set(spans_by_name) == {
        "parent_workflow",
        "subgraph_a",
        "subgraph_b",
    }

    parent_span = spans_by_name["parent_workflow"]
    assert (
        spans_by_name["subgraph_a"].parent.span_id
        == parent_span.context.span_id
    )
    assert (
        spans_by_name["subgraph_b"].parent.span_id
        == parent_span.context.span_id
    )


# ---------------------------------------------------------------------------
# 5. Workflow naming variations (compiled name, run_name, metadata override)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("subgraph_builder", "expected_inner_name"),
    [
        (
            lambda: _graph(_respond, name="compiled_name"),
            "compiled_name",
        ),
        (
            lambda: _graph(_respond).with_config(run_name="config_run_name"),
            "config_run_name",
        ),
        (
            lambda: _graph(_respond, name="compiled_name").with_config(
                {"metadata": {"workflow_name": "metadata_override"}}
            ),
            "metadata_override",
        ),
        (
            lambda: _graph(_respond),
            "LangGraph",
        ),
    ],
    ids=[
        "compiled-name",
        "run-name",
        "metadata-override",
        "default-unnamed",
    ],
)
def test_nested_workflow_naming_variations(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
    subgraph_builder: Any,
    expected_inner_name: str,
) -> None:
    subgraph = subgraph_builder()
    graph = _graph(subgraph, name="outer_graph")

    with instrument_langchain():
        graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2
    inner_span = next(
        span for span in spans if span.name != "invoke_workflow outer_graph"
    )
    assert inner_span.name == f"invoke_workflow {expected_inner_name}"
    assert (
        inner_span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]
        == expected_inner_name
    )


def test_runtime_invoke_config_metadata_override(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    graph = _graph(_respond, name="default_name")

    with instrument_langchain():
        graph.invoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"metadata": {"workflow_name": "runtime_override"}},
        )

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 1
    assert spans[0].name == "invoke_workflow runtime_override"
    assert (
        spans[0].attributes[GenAI.GEN_AI_WORKFLOW_NAME] == "runtime_override"
    )


def test_nested_subgraph_does_not_inherit_parent_workflow_name(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    subgraph = _graph(_respond, name="inner_unique")
    graph = _graph(subgraph, name="outer_unique")

    with instrument_langchain():
        graph.invoke(
            {"messages": [HumanMessage(content="hello")]},
            config={"metadata": {"workflow_name": "outer_override"}},
        )

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2
    inner_span = next(
        span
        for span in spans
        if span.attributes[GenAI.GEN_AI_WORKFLOW_NAME] == "inner_unique"
    )
    assert inner_span.name == "invoke_workflow inner_unique"


def test_parent_and_child_sharing_identical_metadata_values(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    subgraph = _graph(_respond, name="child_workflow").with_config(
        {"metadata": {"workflow_name": "shared_workflow_name"}}
    )
    graph = _graph(subgraph, name="parent_workflow").with_config(
        {"metadata": {"workflow_name": "shared_workflow_name"}}
    )

    with instrument_langchain():
        graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2
    assert all(
        span.attributes.get(GenAI.GEN_AI_WORKFLOW_NAME)
        == "shared_workflow_name"
        for span in spans
    )


def test_nested_agents_sharing_identical_agent_type(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    inner_agent = _graph(_respond, name="inner_agent").with_config(
        {"metadata": {"agent_type": "worker", "agent_name": "inner_worker"}}
    )
    outer_agent = _graph(inner_agent, name="outer_agent").with_config(
        {"metadata": {"agent_type": "worker", "agent_name": "outer_worker"}}
    )

    with instrument_langchain():
        outer_agent.invoke({"messages": [HumanMessage(content="hello")]})

    agent_spans = _agent_spans(span_exporter)
    assert len(agent_spans) == 2
    names = {span.name for span in agent_spans}
    assert names == {"invoke_agent outer_worker", "invoke_agent inner_worker"}


def test_nested_subgraph_runtime_config_metadata_override(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    subgraph = _graph(_respond, name="inner_base")

    def run_inner(state: _State) -> _State:
        return subgraph.invoke(
            state,
            config={"metadata": {"workflow_name": "runtime_inner_override"}},
        )

    outer = _graph(run_inner, name="outer_graph")

    with instrument_langchain():
        outer.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2
    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span for span in spans
    }
    assert "runtime_inner_override" in spans_by_name
    assert "outer_graph" in spans_by_name


# ---------------------------------------------------------------------------
# 6. Workflow / Agent interactions
# ---------------------------------------------------------------------------


def test_workflow_containing_agent(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    agent_subgraph = _graph(_respond, name="agent_worker").with_config(
        {"metadata": {"agent_name": "inner_agent"}}
    )
    workflow_graph = _graph(agent_subgraph, name="outer_workflow")

    with instrument_langchain():
        workflow_graph.invoke({"messages": [HumanMessage(content="hello")]})

    workflow_spans = _workflow_spans(span_exporter)
    agent_spans = _agent_spans(span_exporter)

    assert len(workflow_spans) == 1
    assert workflow_spans[0].name == "invoke_workflow outer_workflow"

    assert len(agent_spans) == 1
    assert agent_spans[0].name == "invoke_agent inner_agent"
    assert agent_spans[0].parent.span_id == workflow_spans[0].context.span_id


def test_agent_containing_workflow(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    calculation_workflow = _graph(_respond, name="calc_workflow")
    agent_graph = _graph(calculation_workflow, name="agent_graph").with_config(
        {"metadata": {"agent_name": "outer_agent"}}
    )

    with instrument_langchain():
        agent_graph.invoke({"messages": [HumanMessage(content="hello")]})

    workflow_spans = _workflow_spans(span_exporter)
    agent_spans = _agent_spans(span_exporter)

    assert len(agent_spans) == 1
    assert agent_spans[0].name == "invoke_agent outer_agent"

    assert len(workflow_spans) == 1
    assert workflow_spans[0].name == "invoke_workflow calc_workflow"
    assert (
        workflow_spans[0].attributes[GenAI.GEN_AI_WORKFLOW_NAME]
        == "calc_workflow"
    )


# ---------------------------------------------------------------------------
# 7. Non-graph runnables inside a workflow should NOT emit workflow spans
# ---------------------------------------------------------------------------


def test_nested_runnables_do_not_emit_workflow_spans(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    sequence = RunnableLambda(_respond) | RunnableLambda(_respond)
    graph = _graph(sequence, name="pipeline_graph")

    with instrument_langchain():
        graph.invoke({"messages": [HumanMessage(content="hello")]})

    workflow_spans = _workflow_spans(span_exporter)
    assert len(workflow_spans) == 1
    assert workflow_spans[0].name == "invoke_workflow pipeline_graph"


# ---------------------------------------------------------------------------
# 8. Error handling in nested workflows (sync invoke & streaming)
# ---------------------------------------------------------------------------


def test_nested_workflow_error_sync(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    failing_subgraph = _graph(_fail_step, name="failing_subgraph")
    graph = _graph(failing_subgraph, name="outer_graph")

    with instrument_langchain():
        with pytest.raises(RuntimeError, match="simulated node failure"):
            graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2

    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span for span in spans
    }
    assert set(spans_by_name) == {"outer_graph", "failing_subgraph"}

    inner_span = spans_by_name["failing_subgraph"]
    outer_span = spans_by_name["outer_graph"]

    assert inner_span.status.status_code == StatusCode.ERROR
    assert inner_span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"

    assert outer_span.status.status_code == StatusCode.ERROR
    assert outer_span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"


def test_nested_workflow_error_streaming(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    failing_subgraph = _graph(_fail_step, name="stream_failing_subgraph")
    graph = _graph(failing_subgraph, name="outer_stream_graph")

    with instrument_langchain():
        with pytest.raises(RuntimeError):
            list(graph.stream({"messages": [HumanMessage(content="hello")]}))

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2

    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes[error_attributes.ERROR_TYPE] == "RuntimeError"


def test_sibling_workflow_partial_failure(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
) -> None:
    subgraph_ok = _graph(_respond, name="healthy_subgraph")
    subgraph_fail = _graph(_fail_step, name="failing_subgraph")

    builder = StateGraph(_State)
    builder.add_node("step_1", subgraph_ok)
    builder.add_node("step_2", subgraph_fail)
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", END)
    graph = builder.compile(name="pipeline")

    with instrument_langchain():
        with pytest.raises(RuntimeError, match="simulated node failure"):
            graph.invoke({"messages": [HumanMessage(content="hello")]})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 3

    spans_by_name = {
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME]: span for span in spans
    }
    assert (
        spans_by_name["healthy_subgraph"].status.status_code
        != StatusCode.ERROR
    )
    assert (
        error_attributes.ERROR_TYPE
        not in spans_by_name["healthy_subgraph"].attributes
    )

    assert (
        spans_by_name["failing_subgraph"].status.status_code
        == StatusCode.ERROR
    )
    assert (
        spans_by_name["failing_subgraph"].attributes[
            error_attributes.ERROR_TYPE
        ]
        == "RuntimeError"
    )

    assert spans_by_name["pipeline"].status.status_code == StatusCode.ERROR
    assert (
        spans_by_name["pipeline"].attributes[error_attributes.ERROR_TYPE]
        == "RuntimeError"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
async def test_nested_runtime_workflow_name_matches_parent(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
    async_mode: bool,
) -> None:
    inner = _graph(_respond, name="inner")
    metadata = {"workflow_name": "shared_workflow"}

    def run_inner(state: _State) -> _State:
        return inner.invoke(state, config={"metadata": metadata})

    async def arun_inner(state: _State) -> _State:
        return await inner.ainvoke(state, config={"metadata": metadata})

    outer = _graph(arun_inner if async_mode else run_inner, name="outer")
    with instrument_langchain():
        inputs = {"messages": [HumanMessage(content="hello")]}
        if async_mode:
            await outer.ainvoke(inputs, config={"metadata": metadata})
        else:
            outer.invoke(inputs, config={"metadata": metadata})

    spans = _workflow_spans(span_exporter)
    assert len(spans) == 2
    assert all(
        span.attributes[GenAI.GEN_AI_WORKFLOW_NAME] == "shared_workflow"
        for span in spans
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("metadata", "expected_name"),
    [
        ({"agent_name": "runtime_agent"}, "invoke_agent runtime_agent"),
        ({"agent_type": "worker"}, "invoke_agent"),
        ({"otel_agent_span": True}, "invoke_agent"),
    ],
)
async def test_runtime_agent_metadata_announces_only_graph_root(
    instrument_langchain: Any,
    span_exporter: InMemorySpanExporter,
    async_mode: bool,
    metadata: dict[str, Any],
    expected_name: str,
) -> None:
    graph = _graph(_respond, name="compiled_graph")

    with instrument_langchain():
        inputs = {"messages": [HumanMessage(content="hello")]}
        config = {"metadata": metadata}
        if async_mode:
            await graph.ainvoke(inputs, config=config)
        else:
            graph.invoke(inputs, config=config)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [expected_name]
