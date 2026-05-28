# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for operation_mapping module.

Tests the public API: classify_chain_run, resolve_agent_name.
"""

import uuid

from opentelemetry.instrumentation.genai.langchain.operation_mapping import (
    OperationName,
    classify_chain_run,
    resolve_agent_name,
)

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
