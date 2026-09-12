# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Focused regression corpus for LangChain create_agent classification."""

from __future__ import annotations

import asyncio
import sys
from importlib import import_module
from typing import Any
from unittest import mock

import langchain.agents
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool
from typing_extensions import Self

import opentelemetry.instrumentation.genai.langchain as langchain_instrumentation
from opentelemetry.instrumentation.genai.langchain import (
    LangChainInstrumentor,
    agent_context,
)
from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    ToolInvocation,
    WorkflowInvocation,
)

create_agent = getattr(langchain.agents, "create_agent", None)
if create_agent is None:
    pytest.skip(
        "create_agent requires a newer langchain version",
        allow_module_level=True,
    )

create_react_agent = import_module("langgraph.prebuilt").create_react_agent
langgraph_graph = import_module("langgraph.graph")
END = langgraph_graph.END
START = langgraph_graph.START
StateGraph = langgraph_graph.StateGraph


class FakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Self:
        return self


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


def test_uninstrument_tolerates_missing_pregel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langgraph_pregel = import_module("langgraph.pregel")
    monkeypatch.delattr(langgraph_pregel, "Pregel")
    monkeypatch.setattr(langchain_instrumentation, "unwrap", mock.Mock())

    LangChainInstrumentor()._uninstrument()


def test_uninstrument_restores_pregel_when_prebuilt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    langgraph_pregel = import_module("langgraph.pregel")
    original_stream = langgraph_pregel.Pregel.stream
    original_astream = langgraph_pregel.Pregel.astream
    LangChainInstrumentor()._instrument_agent_entry_points()
    monkeypatch.setitem(sys.modules, "langgraph.prebuilt", None)

    LangChainInstrumentor()._uninstrument()

    assert langgraph_pregel.Pregel.stream is original_stream
    assert langgraph_pregel.Pregel.astream is original_astream


def _handler() -> tuple[OpenTelemetryLangChainCallbackHandler, mock.MagicMock]:
    telemetry = mock.MagicMock()
    workflow = mock.MagicMock(spec=WorkflowInvocation)
    workflow.span = mock.MagicMock()
    workflow.span.is_recording.return_value = False
    telemetry.workflow.return_value = workflow

    def make_agent(*args: Any, **kwargs: Any) -> mock.MagicMock:
        invocation = mock.MagicMock(spec=AgentInvocation)
        invocation.agent_name = kwargs.get("agent_name")
        invocation.span = mock.MagicMock()
        invocation.span.is_recording.return_value = False
        return invocation

    telemetry.invoke_local_agent.side_effect = make_agent

    tool_invocation = mock.MagicMock(spec=ToolInvocation)
    tool_invocation.span = mock.MagicMock()
    tool_invocation.span.is_recording.return_value = False
    telemetry.tool.return_value = tool_invocation
    return OpenTelemetryLangChainCallbackHandler(telemetry), telemetry


def _agent_names(telemetry: mock.MagicMock) -> list[str]:
    return [
        call.kwargs["agent_name"]
        for call in telemetry.invoke_local_agent.call_args_list
    ]


@pytest.mark.parametrize(
    ("name", "expected_name"),
    [(None, None), ("named_agent", "named_agent")],
)
def test_create_agent_root(
    name: str | None,
    expected_name: str | None,
    span_exporter,
    start_instrumentation,
) -> None:
    agent_kwargs = {"name": name} if name is not None else {}
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        **agent_kwargs,
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    expected_span_name = (
        f"invoke_agent {expected_name}" if expected_name else "invoke_agent"
    )
    assert [span.name for span in spans] == [expected_span_name]
    agent_span = spans[0]
    assert agent_span.parent is None
    if expected_name:
        assert agent_span.attributes["gen_ai.agent.name"] == expected_name
    else:
        assert "gen_ai.agent.name" not in agent_span.attributes


def test_create_agent_stream_root(
    span_exporter, start_instrumentation
) -> None:
    stream = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="stream_agent",
    ).stream({"messages": [("user", "hi")]})

    for _ in stream:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent stream_agent"]
    assert spans[0].parent is None
    assert spans[0].attributes["gen_ai.agent.name"] == "stream_agent"
    assert agent_context._pending.get() == ()


@pytest.mark.asyncio
async def test_create_agent_astream_root(
    span_exporter, start_instrumentation
) -> None:
    stream = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="astream_agent",
    ).astream({"messages": [("user", "hi")]})

    async for _ in stream:
        pass

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent astream_agent"]
    assert spans[0].parent is None
    assert spans[0].attributes["gen_ai.agent.name"] == "astream_agent"
    assert agent_context._pending.get() == ()


class _MarkedAgentGraph:
    config = {"metadata": {"ls_integration": "langchain_create_agent"}}


def _wrapped_stream(source: Any) -> Any:
    return agent_context.wrap_stream(
        lambda: source, _MarkedAgentGraph(), (), {}
    )


def _wrapped_astream(source: Any) -> Any:
    return agent_context.wrap_astream(
        lambda: source, _MarkedAgentGraph(), (), {}
    )


def test_stream_exception_before_first_item_withdraws_announcement() -> None:
    _, telemetry = _handler()

    def failing_stream():
        raise RuntimeError("before first item")
        yield

    stream = _wrapped_stream(failing_stream())
    with pytest.raises(RuntimeError, match="before first item"):
        next(stream)

    assert agent_context._pending.get() == ()
    telemetry.invoke_local_agent.assert_not_called()


def test_stream_abandoned_before_start_does_not_announce() -> None:
    stream = _wrapped_stream(iter((1,)))

    assert agent_context._pending.get() == ()
    stream.close()
    assert agent_context._pending.get() == ()


def test_stream_close_after_first_item_withdraws_announcement() -> None:
    stream = _wrapped_stream(iter((1, 2)))

    next(stream)
    assert agent_context._pending.get() == ()
    stream.close()
    assert agent_context._pending.get() == ()


@pytest.mark.asyncio
async def test_astream_exception_before_first_item_withdraws_announcement() -> (
    None
):
    _, telemetry = _handler()

    async def failing_stream():
        raise RuntimeError("before first item")
        yield

    stream = _wrapped_astream(failing_stream())
    with pytest.raises(RuntimeError, match="before first item"):
        await stream.__anext__()

    assert agent_context._pending.get() == ()
    telemetry.invoke_local_agent.assert_not_called()


@pytest.mark.asyncio
async def test_astream_abandoned_before_start_does_not_announce() -> None:
    async def source():
        yield 1

    stream = _wrapped_astream(source())

    assert agent_context._pending.get() == ()
    await stream.aclose()
    assert agent_context._pending.get() == ()


@pytest.mark.asyncio
async def test_astream_close_after_first_item_withdraws_announcement() -> None:
    async def source():
        yield 1
        yield 2

    stream = _wrapped_astream(source())

    await stream.__anext__()
    assert agent_context._pending.get() == ()
    await stream.aclose()
    assert agent_context._pending.get() == ()


def test_unnamed_create_agent_has_no_placeholder_name(
    span_exporter, start_instrumentation
) -> None:
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent"]
    assert "gen_ai.agent.name" not in spans[0].attributes


def test_preimported_create_react_agent_is_classified_at_runtime(
    span_exporter, start_instrumentation
) -> None:
    create_react_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="react_agent",
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["invoke_agent react_agent"]
    assert spans[0].attributes["gen_ai.agent.name"] == "react_agent"


def test_create_agent_name_wins_over_run_name_override(
    span_exporter, start_instrumentation
) -> None:
    agent = create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="planner_agent",
    )
    RunnableLambda(
        lambda _: agent.invoke(
            {"messages": [("user", "hi")]}, {"run_name": "step1"}
        )
    ).with_config(run_name="planner").invoke({})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "invoke_agent planner_agent",
        "invoke_workflow planner",
    ]
    agent_span, workflow_span = spans
    assert agent_span.parent.span_id == workflow_span.context.span_id
    assert workflow_span.parent is None
    assert agent_span.attributes["gen_ai.agent.name"] == "planner_agent"
    assert "gen_ai.agent.name" not in workflow_span.attributes


def test_create_agent_internal_nodes_are_not_agents(
    span_exporter, start_instrumentation
) -> None:
    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
        name="agent",
    ).invoke({"messages": [("user", "hi")]})

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "execute_tool noop",
        "invoke_agent agent",
    ]
    agent_span = spans[-1]
    assert agent_span.parent is None
    assert agent_span.attributes["gen_ai.agent.name"] == "agent"
    tool_span = spans[0]
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert tool_span.attributes["gen_ai.agent.name"] == "agent"


def test_create_agent_with_configured_agent_name_emits_one_agent() -> None:
    handler, telemetry = _handler()
    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "noop", "args": {}, "id": "1"}],
                ),
                AIMessage(content="done"),
            ]
        ),
        [noop],
    ).with_config(metadata={"agent_name": "ordinary"}).invoke(
        {"messages": [("user", "hi")]}, {"callbacks": [handler]}
    )

    assert _agent_names(telemetry) == ["ordinary"]
    assert telemetry.tool.call_count == 1


def test_ordinary_runnable_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    RunnableLambda(lambda value: value).with_config(
        run_name="ordinary"
    ).invoke("value", {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


def test_plain_state_graph_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    builder = StateGraph(dict[str, int])
    builder.add_node("increment", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    builder.compile().invoke({"value": 1}, {"callbacks": [handler]})

    telemetry.invoke_local_agent.assert_not_called()


def test_plain_state_graph_with_agent_node_is_not_an_agent() -> None:
    handler, telemetry = _handler()
    builder = StateGraph(dict[str, int])
    builder.add_node("agent", lambda state: {"value": state["value"] + 1})
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    builder.compile(name="support_pipeline").invoke(
        {"value": 1}, {"callbacks": [handler]}
    )

    telemetry.invoke_local_agent.assert_not_called()


def _span_named(spans: list[Any], name: str) -> Any:
    matches = [span for span in spans if span.name == name]
    assert len(matches) == 1, [
        (
            span.name,
            span.context.span_id,
            span.parent.span_id if span.parent else None,
        )
        for span in spans
    ]
    return matches[0]


def _root_span_named(spans: list[Any], name: str) -> Any:
    matches = [
        span for span in spans if span.name == name and span.parent is None
    ]
    assert len(matches) == 1
    return matches[0]


def _assert_parent(child: Any, parent: Any) -> None:
    assert child.parent.span_id == parent.context.span_id


def test_nested_agent_emits_no_extra_agent_spans(
    span_exporter, start_instrumentation
) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def delegate(config: RunnableConfig) -> str:
        """Delegate to the inner agent."""
        result = inner.invoke({"messages": [("user", "work")]}, config)
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate", "args": {}, "id": "call-1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate],
        name="outer",
    ).with_config(metadata={"agent_name": "math_agent"})
    outer.invoke({"messages": [("user", "start")]})

    spans = span_exporter.get_finished_spans()
    assert sorted(span.name for span in spans) == [
        "execute_tool delegate",
        "invoke_agent inner",
        "invoke_agent math_agent",
    ]
    outer_span = _root_span_named(spans, "invoke_agent math_agent")
    tool_span = _span_named(spans, "execute_tool delegate")
    inner_span = _span_named(spans, "invoke_agent inner")
    assert outer_span.parent is None
    _assert_parent(tool_span, outer_span)
    _assert_parent(inner_span, tool_span)


def test_agent_metadata_rename_is_preserved(
    span_exporter, start_instrumentation
) -> None:
    create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="original",
    ).with_config(metadata={"agent_name": "renamed"}).invoke(
        {"messages": [("user", "hi")]}
    )

    spans = span_exporter.get_finished_spans()
    renamed_span = _span_named(spans, "invoke_agent renamed")
    assert renamed_span.attributes["gen_ai.agent.name"] == "renamed"


def test_nested_named_agent_uses_its_declared_name(
    span_exporter, start_instrumentation
) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def delegate_named(config: RunnableConfig) -> str:
        """Delegate to the named inner agent."""
        result = inner.invoke({"messages": [("user", "work")]}, config)
        return str(result["messages"][-1].content)

    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate_named", "args": {}, "id": "1"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [delegate_named],
        name="outer",
    ).with_config(metadata={"agent_name": "renamed_outer"}).invoke(
        {"messages": [("user", "start")]}
    )

    spans = span_exporter.get_finished_spans()
    outer_span = _root_span_named(spans, "invoke_agent renamed_outer")
    inner_span = _span_named(spans, "invoke_agent inner")
    tool_span = _span_named(spans, "execute_tool delegate_named")
    _assert_parent(tool_span, outer_span)
    _assert_parent(inner_span, tool_span)


def test_three_level_agents_resolve_names_against_all_ancestors(
    span_exporter, start_instrumentation
) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def to_inner(config: RunnableConfig) -> str:
        """Invoke the inner agent."""
        result = inner.invoke({"messages": [("user", "inner work")]}, config)
        return str(result["messages"][-1].content)

    middle = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "to_inner", "args": {}, "id": "2"}],
                ),
                AIMessage(content="middle done"),
            ]
        ),
        [to_inner],
        name="middle",
    )

    @tool
    def to_middle(config: RunnableConfig) -> str:
        """Invoke the middle agent."""
        result = middle.invoke({"messages": [("user", "middle work")]}, config)
        return str(result["messages"][-1].content)

    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "to_middle", "args": {}, "id": "3"}],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [to_middle],
        name="outer",
    ).with_config(metadata={"agent_name": "renamed_outer"}).invoke(
        {"messages": [("user", "start")]}
    )

    spans = span_exporter.get_finished_spans()
    outer_span = _root_span_named(spans, "invoke_agent renamed_outer")
    outer_tool = _span_named(spans, "execute_tool to_middle")
    middle_span = _span_named(spans, "invoke_agent middle")
    middle_tool = _span_named(spans, "execute_tool to_inner")
    inner_span = _span_named(spans, "invoke_agent inner")
    _assert_parent(outer_tool, outer_span)
    _assert_parent(middle_span, outer_tool)
    _assert_parent(inner_span, middle_tool)


@pytest.mark.asyncio
async def test_async_root_agent(span_exporter, start_instrumentation) -> None:
    await create_agent(
        FakeModel(responses=[AIMessage(content="done")]),
        [noop],
        name="async_root",
    ).ainvoke({"messages": [("user", "hi")]})

    span = _span_named(
        span_exporter.get_finished_spans(), "invoke_agent async_root"
    )
    assert span.parent is None


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The LangChain async callback path does not propagate the context "
        "token attached at span start, so spans are emitted without parentage. "
        "Pre-existing behavior, not introduced by this change."
    ),
)
@pytest.mark.asyncio
async def test_async_nested_agent(
    span_exporter, start_instrumentation
) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="async_inner",
    )

    @tool
    async def async_delegate(config: RunnableConfig) -> str:
        """Delegate asynchronously."""
        result = await inner.ainvoke({"messages": [("user", "work")]}, config)
        return str(result["messages"][-1].content)

    outer = create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "async_delegate", "args": {}, "id": "4"}
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [async_delegate],
        name="async_outer",
    )
    await outer.ainvoke({"messages": [("user", "start")]})

    spans = span_exporter.get_finished_spans()
    outer_span = _root_span_named(spans, "invoke_agent async_outer")
    tool_span = _span_named(spans, "execute_tool async_delegate")
    inner_span = _span_named(spans, "invoke_agent async_inner")
    assert outer_span.parent is None
    _assert_parent(tool_span, outer_span)
    _assert_parent(inner_span, tool_span)


@pytest.mark.asyncio
async def test_concurrent_async_agents_have_distinct_roots(
    span_exporter, start_instrumentation
) -> None:
    first = create_agent(
        FakeModel(responses=[AIMessage(content="first done")]),
        [noop],
        name="first_agent",
    )
    second = create_agent(
        FakeModel(responses=[AIMessage(content="second done")]),
        [noop],
        name="second_agent",
    )
    await asyncio.gather(
        first.ainvoke({"messages": [("user", "first")]}),
        second.ainvoke({"messages": [("user", "second")]}),
    )

    spans = span_exporter.get_finished_spans()
    first_span = _span_named(spans, "invoke_agent first_agent")
    second_span = _span_named(spans, "invoke_agent second_agent")
    assert first_span.parent is None
    assert second_span.parent is None


def test_announced_root_bypasses_inherited_middleware_name(
    span_exporter, start_instrumentation
) -> None:
    inner = create_agent(
        FakeModel(responses=[AIMessage(content="inner done")]),
        [noop],
        name="inner",
    )

    @tool
    def middleware_delegate(config: RunnableConfig) -> str:
        """Invoke an agent below middleware-style metadata."""
        result = inner.invoke({"messages": [("user", "work")]}, config)
        return str(result["messages"][-1].content)

    create_agent(
        FakeModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "middleware_delegate",
                            "args": {},
                            "id": "5",
                        }
                    ],
                ),
                AIMessage(content="outer done"),
            ]
        ),
        [middleware_delegate],
        name="outer",
    ).with_config(metadata={"agent_name": "Middleware.parent"}).invoke(
        {"messages": [("user", "start")]}
    )

    spans = span_exporter.get_finished_spans()
    outer_span = _root_span_named(spans, "invoke_agent Middleware.parent")
    tool_span = _span_named(spans, "execute_tool middleware_delegate")
    inner_span = _span_named(spans, "invoke_agent inner")
    _assert_parent(tool_span, outer_span)
    _assert_parent(inner_span, tool_span)
