# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for operation_mapping module.

Tests the public API: classify_chain_run, resolve_agent_name.
"""

from __future__ import annotations

import uuid
from typing import Any

import langchain.agents
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from typing_extensions import Self

from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    OperationName,
    classify_chain_run,
    resolve_agent_name,
)


class _FakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Self:
        return self


@tool
def _noop() -> str:
    """Do nothing."""
    return "ok"


def _create_agent(*args: Any, **kwargs: Any) -> Any:
    create_agent = getattr(langchain.agents, "create_agent", None)
    if create_agent is None:
        pytest.skip("create_agent requires a newer langchain version")
    return create_agent(*args, **kwargs)


# ---------------------------------------------------------------------------
# resolve_agent_name
# ---------------------------------------------------------------------------


class TestResolveAgentName:
    def test_metadata_agent_name_takes_highest_priority(self):
        result = resolve_agent_name(
            serialized={"name": "serialized_name"},
            metadata={"agent_name": "meta_name"},
            kwargs={"name": "kwargs_name"},
        )
        assert result == "meta_name"

    def test_kwargs_name_used_when_no_metadata_agent_name(self):
        result = resolve_agent_name(
            serialized={"name": "serialized_name"},
            metadata={},
            kwargs={"name": "kwargs_name"},
        )
        assert result == "kwargs_name"

    def test_serialized_name_used_as_fallback(self):
        result = resolve_agent_name(
            serialized={"name": "serialized_name"},
            metadata={},
            kwargs={},
        )
        assert result == "serialized_name"

    def test_langgraph_node_used_as_last_resort(self):
        result = resolve_agent_name(
            serialized={},
            metadata={"langgraph_node": "my_node"},
            kwargs={},
        )
        assert result == "my_node"

    def test_langgraph_start_node_not_returned(self):
        result = resolve_agent_name(
            serialized={},
            metadata={"langgraph_node": "__start__"},
            kwargs={},
        )
        assert result is None

    def test_returns_none_when_nothing_available(self):
        result = resolve_agent_name(
            serialized={},
            metadata=None,
            kwargs={},
        )
        assert result is None

    def test_none_metadata_falls_through_to_serialized(self):
        result = resolve_agent_name(
            serialized={"name": "from_serialized"},
            metadata=None,
            kwargs={},
        )
        assert result == "from_serialized"

    def test_result_is_always_str(self):
        # metadata value that is not already a string
        result = resolve_agent_name(
            serialized={},
            metadata={"agent_name": 42},
            kwargs={},
        )
        assert result == "42"
        assert isinstance(result, str)

    def test_metadata_rename_wins_without_ancestors(self):
        result = resolve_agent_name(
            serialized={},
            metadata={"agent_name": "renamed"},
            kwargs={},
            declared_agent_name="declared",
            ancestor_agent_names=set(),
        )
        assert result == "renamed"

    def test_inherited_metadata_name_falls_through_to_declared_name(self):
        result = resolve_agent_name(
            serialized={},
            metadata={"agent_name": "outer"},
            kwargs={},
            declared_agent_name="inner",
            ancestor_agent_names={"outer", "middle"},
        )
        assert result == "inner"

    def test_inherited_grandparent_name_falls_through_to_declared_name(self):
        result = resolve_agent_name(
            serialized={},
            metadata={"agent_name": "OUTER"},
            kwargs={},
            declared_agent_name="inner",
            ancestor_agent_names={"outer", "middle"},
        )
        assert result == "inner"


# ---------------------------------------------------------------------------
# classify_chain_run
# ---------------------------------------------------------------------------


class TestClassifyChainRun:
    # --- invoke_workflow ---

    def test_langgraph_name_at_root_is_workflow(self):
        result = classify_chain_run(
            serialized={"name": "LangGraph"},
            metadata=None,
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_WORKFLOW

    def test_langgraph_in_graph_id_at_root_is_workflow(self):
        result = classify_chain_run(
            serialized={"name": "MyGraph", "graph": {"id": "LangGraph-abc"}},
            metadata=None,
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_WORKFLOW

    def test_explicit_workflow_override_at_root(self):
        result = classify_chain_run(
            serialized={"name": "SomeName"},
            metadata={"otel_workflow_span": True},
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_WORKFLOW

    def test_explicit_workflow_override_with_parent(self):
        result = classify_chain_run(
            serialized={"name": "SomeName"},
            metadata={"otel_workflow_span": True},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_WORKFLOW

    @pytest.mark.parametrize(
        ("metadata", "announced_workflow", "expected"),
        [
            pytest.param(
                {"agent_type": "outer"},
                True,
                OperationName.INVOKE_WORKFLOW,
                id="announcement-beats-inherited-agent-type",
            ),
            pytest.param(
                {"otel_agent_span": True},
                True,
                OperationName.INVOKE_WORKFLOW,
                id="announcement-beats-inherited-agent-override",
            ),
            pytest.param(
                {
                    "otel_workflow_span": True,
                    "agent_type": "inherited",
                },
                False,
                OperationName.INVOKE_WORKFLOW,
                id="explicit-workflow-beats-agent-type",
            ),
            pytest.param(
                {
                    "otel_workflow_span": True,
                    "otel_agent_span": False,
                },
                False,
                OperationName.INVOKE_WORKFLOW,
                id="explicit-workflow-beats-agent-suppression",
            ),
        ],
    )
    def test_workflow_override_precedence(
        self,
        metadata: dict[str, Any],
        announced_workflow: bool,
        expected: str,
    ) -> None:
        result = classify_chain_run(
            serialized={},
            metadata=metadata,
            kwargs={"name": "named_subgraph"},
            parent_run_id=uuid.uuid4(),
            announced_workflow=announced_workflow,
        )
        assert result == expected

    def test_root_graph_announcement_beats_inherited_agent_metadata(self):
        result = classify_chain_run(
            serialized={},
            metadata={"agent_type": "outer"},
            kwargs={"name": "LangGraph"},
            parent_run_id=None,
            announced_workflow=True,
        )
        assert result == OperationName.INVOKE_WORKFLOW

    def test_root_chain_with_no_signals_is_workflow(self):
        # A root chain (no parent) with no special names defaults to workflow.
        result = classify_chain_run(
            serialized={},
            metadata=None,
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_WORKFLOW

    def test_langgraph_name_with_parent_is_not_workflow(self):
        # Having a parent disqualifies it from being a top-level workflow.
        result = classify_chain_run(
            serialized={"name": "LangGraph"},
            metadata=None,
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        # Not a workflow; no agent signals → suppressed
        assert result is None

    # --- invoke_agent ---

    def test_agent_name_metadata_is_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={"agent_name": "my_agent"},
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_AGENT

    def test_agent_type_metadata_is_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={"agent_type": "react"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    def test_otel_agent_span_true_is_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={"otel_agent_span": True},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    def test_internal_langgraph_node_is_not_agent(
        self, span_exporter, start_instrumentation
    ):
        _create_agent(
            _FakeModel(responses=[AIMessage(content="done")]),
            [_noop],
            name="my_agent",
        ).invoke({"messages": [("user", "hi")]})

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == ["invoke_agent my_agent"]
        agent_span = spans[0]
        assert agent_span.parent is None
        assert agent_span.attributes["gen_ai.agent.name"] == "my_agent"

    def test_nested_agent_gets_its_own_span_under_the_calling_tool(
        self, span_exporter, start_instrumentation
    ):
        inner = _create_agent(
            _FakeModel(responses=[AIMessage(content="inner done")]),
            [_noop],
            name="inner_agent",
        )

        @tool
        def delegate(config: RunnableConfig) -> str:
            """Delegate to the inner agent."""
            result = inner.invoke({"messages": [("user", "work")]}, config)
            return str(result["messages"][-1].content)

        outer = _create_agent(
            _FakeModel(
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
            name="outer_agent",
        )
        outer.invoke({"messages": [("user", "start")]})

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "invoke_agent inner_agent",
            "execute_tool delegate",
            "invoke_agent outer_agent",
        ]
        inner_agent, tool_span, outer_agent = spans
        assert outer_agent.parent is None
        assert tool_span.parent.span_id == outer_agent.context.span_id
        assert inner_agent.parent.span_id == tool_span.context.span_id
        assert inner_agent.attributes["gen_ai.agent.name"] == "inner_agent"
        assert outer_agent.attributes["gen_ai.agent.name"] == "outer_agent"
        assert tool_span.attributes["gen_ai.agent.name"] == "outer_agent"

    def test_unnamed_nested_agent_has_no_placeholder_name(
        self, span_exporter, start_instrumentation
    ):
        inner = _create_agent(
            _FakeModel(responses=[AIMessage(content="inner done")]),
            [_noop],
        )

        @tool
        def delegate(config: RunnableConfig) -> str:
            """Delegate to the unnamed inner agent."""
            result = inner.invoke({"messages": [("user", "work")]}, config)
            return str(result["messages"][-1].content)

        outer = _create_agent(
            _FakeModel(
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
            name="outer_agent",
        )
        outer.invoke({"messages": [("user", "start")]})

        spans = span_exporter.get_finished_spans()
        assert [span.name for span in spans] == [
            "invoke_agent",
            "execute_tool delegate",
            "invoke_agent outer_agent",
        ]
        inner_agent, tool_span, outer_agent = spans
        assert outer_agent.parent is None
        assert tool_span.parent.span_id == outer_agent.context.span_id
        assert inner_agent.parent.span_id == tool_span.context.span_id
        assert "gen_ai.agent.name" not in inner_agent.attributes
        assert outer_agent.attributes["gen_ai.agent.name"] == "outer_agent"
        assert tool_span.attributes["gen_ai.agent.name"] == "outer_agent"

    def test_explicit_otel_agent_metadata_overrides_node_inference(self):
        result = classify_chain_run(
            serialized={},
            metadata={
                "agent_type": "react",
                "lc_agent_name": "my_agent",
                "langgraph_node": "model",
            },
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    def test_langgraph_node_metadata_with_parent_is_suppressed(self):
        # langgraph_node alone is no longer an agent signal in _has_agent_signals;
        # it is only used by resolve_agent_name for name resolution.
        # A child chain with only langgraph_node metadata and no other agent
        # signals (otel_agent_span, agent_name, agent_type) is suppressed.
        result = classify_chain_run(
            serialized={},
            metadata={"langgraph_node": "my_node"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result is None

    def test_langgraph_node_with_agent_name_is_agent(self):
        # langgraph_node combined with agent_name still produces INVOKE_AGENT
        # because agent_name triggers _has_agent_signals.
        result = classify_chain_run(
            serialized={},
            metadata={"langgraph_node": "my_node", "agent_name": "my_agent"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    # Agent signals take priority over workflow signals.
    def test_agent_signals_beat_workflow_signals(self):
        result = classify_chain_run(
            serialized={"name": "LangGraph"},
            metadata={"agent_name": "my_agent"},
            kwargs={},
            parent_run_id=None,
        )
        assert result == OperationName.INVOKE_AGENT

    # --- suppressed ---

    def test_start_node_is_suppressed(self):
        result = classify_chain_run(
            serialized={},
            metadata={"langgraph_node": "__start__"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result is None

    def test_otel_trace_false_is_suppressed(self):
        result = classify_chain_run(
            serialized={"name": "LangGraph"},
            metadata={"otel_trace": False},
            kwargs={},
            parent_run_id=None,
        )
        assert result is None

    def test_middleware_name_is_suppressed(self):
        result = classify_chain_run(
            serialized={"name": "Middleware.Router"},
            metadata=None,
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result is None

    def test_otel_agent_span_false_with_no_other_signals_suppressed(self):
        result = classify_chain_run(
            serialized={},
            metadata={"otel_agent_span": False},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result is None

    def test_otel_agent_span_false_suppresses_inferred_langchain_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={
                "otel_agent_span": False,
                "lc_agent_name": "my_agent",
                "ls_integration": "langchain_create_agent",
            },
            kwargs={},
            parent_run_id=None,
        )
        assert result is None

    def test_otel_agent_span_false_with_agent_name_is_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={"otel_agent_span": False, "agent_name": "my_agent"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    def test_otel_agent_span_false_with_agent_type_is_agent(self):
        result = classify_chain_run(
            serialized={},
            metadata={"otel_agent_span": False, "agent_type": "react"},
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result == OperationName.INVOKE_AGENT

    def test_non_langgraph_child_chain_suppressed(self):
        # Child chain with no agent or workflow signals → suppressed.
        result = classify_chain_run(
            serialized={"name": "SomeInternalChain"},
            metadata=None,
            kwargs={},
            parent_run_id=uuid.uuid4(),
        )
        assert result is None
