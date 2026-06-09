# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for on_chain_start / on_chain_end / on_chain_error in
OpenTelemetryLangChainCallbackHandler.

All TelemetryHandler interactions are mocked so that these tests exercise only
the callback-handler logic and the invocation-manager bookkeeping.
"""

import uuid
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    make_input_message,
    make_last_output_message,
    make_output_message,
    serialize,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import InputMessage, OutputMessage, Text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent_inv_mock() -> mock.MagicMock:
    """Return a spec'd AgentInvocation mock with agent_name pre-configured."""
    agent_inv = mock.MagicMock(spec=AgentInvocation)
    agent_inv.span = mock.MagicMock()
    agent_inv.span.is_recording.return_value = False
    # agent_name is an instance attribute set in AgentInvocation.__init__ via the
    # constructor arg; pre-configure it so spec-restricted attribute access works.
    agent_inv.agent_name = None
    return agent_inv


def _make_invoke_local_agent_side_effect(inv: mock.MagicMock):
    """Return a side_effect for invoke_local_agent that mirrors what the real
    AgentInvocation constructor does: set agent_name from the kwarg."""

    def _side_effect(*args, **kwargs):
        inv.agent_name = kwargs.get("agent_name")
        return inv

    return _side_effect


def _make_handler():
    """Return a handler wired to a MagicMock TelemetryHandler."""
    telemetry = mock.MagicMock()

    # workflow returns a mock WorkflowInvocation
    workflow_inv = mock.MagicMock(spec=WorkflowInvocation)
    workflow_inv.span = mock.MagicMock()
    workflow_inv.span.is_recording.return_value = False
    telemetry.workflow.return_value = workflow_inv

    # invoke_local_agent returns a mock AgentInvocation whose agent_name is set
    # to match whatever agent_name kwarg was passed (mirrors real constructor).
    agent_inv = _make_agent_inv_mock()
    telemetry.invoke_local_agent.side_effect = (
        _make_invoke_local_agent_side_effect(agent_inv)
    )

    handler = OpenTelemetryLangChainCallbackHandler(telemetry)
    return handler, telemetry, workflow_inv, agent_inv


def _run_id():
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# on_chain_start – INVOKE_WORKFLOW
# ---------------------------------------------------------------------------


class TestOnChainStartWorkflow:
    def test_workflow_span_created(self):
        handler, telemetry, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        # LangGraph graph serialized dict triggers workflow classification
        handler.on_chain_start(
            serialized={"name": "LangGraph", "id": ["langgraph"]},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        telemetry.workflow.assert_called_once()
        assert (
            handler._invocation_manager.get_invocation(run_id) is workflow_inv
        )

    def test_workflow_name_from_serialized(self):
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "MyLangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        telemetry.workflow.assert_called_once_with(name="MyLangGraph")

    def test_workflow_name_overridden_by_metadata(self):
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "MyLangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"workflow_name": "custom_workflow"},
        )

        telemetry.workflow.assert_called_once_with(name="custom_workflow")

    def test_workflow_registered_in_invocation_manager(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        assert (
            handler._invocation_manager.get_invocation(run_id) is workflow_inv
        )


# ---------------------------------------------------------------------------
# on_chain_start – INVOKE_AGENT
# ---------------------------------------------------------------------------


class TestOnChainStartAgent:
    def test_new_agent_span_created(self):
        handler, telemetry, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "ls_provider": "openai"},
        )

        telemetry.invoke_local_agent.assert_called_once_with(
            provider="openai",
            agent_name="math_agent",
        )
        assert agent_inv.agent_name == "math_agent"
        assert handler._invocation_manager.get_invocation(run_id) is agent_inv

    def test_agent_metadata_set(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={
                "agent_name": "math_agent",
                "agent_id": "agent-123",
                "agent_description": "does math",
                "thread_id": "thread-abc",
            },
        )

        assert agent_inv.agent_id == "agent-123"
        assert agent_inv.agent_description == "does math"
        assert agent_inv.conversation_id == "thread-abc"

    def test_conversation_id_prefers_thread_id_over_session_id(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={
                "agent_name": "math_agent",
                "thread_id": "t1",
                "session_id": "s1",
            },
        )

        assert agent_inv.conversation_id == "t1"

    def test_duplicate_agent_name_does_not_create_new_span(self):
        """When the nearest ancestor already has the same agent name, no new
        AgentInvocation span is created; the run is still tracked with None."""
        handler, telemetry, _, agent_inv = _make_handler()
        parent_run_id = _run_id()
        child_run_id = _run_id()

        # Register the parent agent
        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=parent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )
        telemetry.invoke_local_agent.reset_mock()

        # A child chain with the same agent name should NOT create a new span
        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            metadata={"agent_name": "math_agent"},
        )

        telemetry.invoke_local_agent.assert_not_called()
        assert handler._invocation_manager.get_invocation(child_run_id) is None

    def test_different_agent_name_creates_new_span(self):
        """A child chain with a different agent name creates a new AgentInvocation."""
        handler, telemetry, _, _ = _make_handler()
        parent_run_id = _run_id()
        child_run_id = _run_id()

        # First agent
        first_agent_inv = _make_agent_inv_mock()
        telemetry.invoke_local_agent.side_effect = (
            _make_invoke_local_agent_side_effect(first_agent_inv)
        )

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=parent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        # Second agent with a different name
        second_agent_inv = _make_agent_inv_mock()
        telemetry.invoke_local_agent.side_effect = (
            _make_invoke_local_agent_side_effect(second_agent_inv)
        )

        handler.on_chain_start(
            serialized={"name": "weather_agent"},
            inputs={},
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            metadata={"agent_name": "weather_agent"},
        )

        assert (
            handler._invocation_manager.get_invocation(child_run_id)
            is second_agent_inv
        )
        assert second_agent_inv.agent_name == "weather_agent"

    def test_agent_name_comparison_is_case_insensitive(self):
        handler, telemetry, _, _ = _make_handler()
        parent_run_id = _run_id()
        child_run_id = _run_id()

        parent_agent_inv = _make_agent_inv_mock()
        telemetry.invoke_local_agent.side_effect = (
            _make_invoke_local_agent_side_effect(parent_agent_inv)
        )

        handler.on_chain_start(
            serialized={"name": "Math_Agent"},
            inputs={},
            run_id=parent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "Math_Agent"},
        )
        telemetry.invoke_local_agent.reset_mock()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=child_run_id,
            parent_run_id=parent_run_id,
            metadata={"agent_name": "math_agent"},
        )

        # Same name (case-insensitive) → no new span
        telemetry.invoke_local_agent.assert_not_called()

    def test_no_agent_name_registers_none_invocation(self):
        """When resolve_agent_name returns None the run_id must still be
        registered so that child traversal works."""
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()

        # metadata has otel_agent_span=True so classify_chain_run → INVOKE_AGENT,
        # but no agent_name / kwargs name / serialized name, so resolve_agent_name
        # returns None.
        handler.on_chain_start(
            serialized={},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"otel_agent_span": True},
        )

        telemetry.invoke_local_agent.assert_not_called()
        # run_id must still be registered (with None invocation) so traversal works
        assert run_id in handler._invocation_manager._invocations
        assert handler._invocation_manager.get_invocation(run_id) is None

    def test_no_agent_name_child_can_still_find_ancestor_agent(self):
        """Even when an intermediate node has no agent name, a deeper child
        must still be able to walk up and find a grandparent AgentInvocation."""
        handler, telemetry, _, agent_inv = _make_handler()
        grandparent_id = _run_id()
        parent_id = _run_id()

        # Grandparent: a known agent
        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=grandparent_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        # Parent: INVOKE_AGENT but no resolvable name → registers None
        telemetry.invoke_local_agent.reset_mock()
        handler.on_chain_start(
            serialized={},
            inputs={},
            run_id=parent_id,
            parent_run_id=grandparent_id,
            metadata={"otel_agent_span": True},
        )

        # Child: should find the grandparent agent via _find_nearest_agent
        found = handler._find_nearest_agent(parent_id)
        assert found is agent_inv


# ---------------------------------------------------------------------------
# on_chain_start – unclassified
# ---------------------------------------------------------------------------


class TestOnChainStartUnclassified:
    def test_unclassified_chain_registers_none_and_no_span(self):
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()
        parent_run_id = _run_id()

        # Register the parent first so that the child links correctly
        handler._invocation_manager.add_invocation_state(
            parent_run_id, None, None
        )

        handler.on_chain_start(
            serialized={"name": "SomeInternalChain"},
            inputs={},
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        telemetry.workflow.assert_not_called()
        telemetry.invoke_local_agent.assert_not_called()
        assert run_id in handler._invocation_manager._invocations
        assert handler._invocation_manager.get_invocation(run_id) is None


# ---------------------------------------------------------------------------
# on_chain_end
# ---------------------------------------------------------------------------


class TestOnChainEnd:
    def test_workflow_invocation_stopped_on_chain_end(self):
        handler, telemetry, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_end(outputs={}, run_id=run_id)

        workflow_inv.stop.assert_called_once()

    def test_agent_invocation_stopped_on_chain_end(self):
        handler, telemetry, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        handler.on_chain_end(outputs={}, run_id=run_id)

        agent_inv.stop.assert_called_once()

    def test_none_invocation_on_chain_end_does_not_raise(self):
        """on_chain_end for a run registered with None invocation (unclassified
        or duplicate agent) must silently do nothing."""
        handler, _, _, _ = _make_handler()
        run_id = _run_id()

        handler._invocation_manager.add_invocation_state(run_id, None, None)

        # Must not raise
        handler.on_chain_end(outputs={}, run_id=run_id)

    def test_unknown_run_id_on_chain_end_does_not_raise(self):
        handler, _, _, _ = _make_handler()
        handler.on_chain_end(outputs={}, run_id=_run_id())

    def test_invocation_state_cleaned_up_after_chain_end(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        # span.is_recording() returns False → should be cleaned up
        workflow_inv.span.is_recording.return_value = False
        handler.on_chain_end(outputs={}, run_id=run_id)

        assert run_id not in handler._invocation_manager._invocations


# ---------------------------------------------------------------------------
# on_chain_error
# ---------------------------------------------------------------------------


class TestOnChainError:
    def test_workflow_invocation_failed_on_chain_error(self):
        handler, telemetry, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        err = RuntimeError("something went wrong")
        handler.on_chain_error(error=err, run_id=run_id)

        workflow_inv.fail.assert_called_once_with(err)

    def test_agent_invocation_failed_on_chain_error(self):
        handler, telemetry, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        err = ValueError("agent failed")
        handler.on_chain_error(error=err, run_id=run_id)

        agent_inv.fail.assert_called_once_with(err)

    def test_none_invocation_on_chain_error_does_not_raise(self):
        handler, _, _, _ = _make_handler()
        run_id = _run_id()

        handler._invocation_manager.add_invocation_state(run_id, None, None)

        handler.on_chain_error(error=RuntimeError("boom"), run_id=run_id)

    def test_unknown_run_id_on_chain_error_does_not_raise(self):
        handler, _, _, _ = _make_handler()
        handler.on_chain_error(error=RuntimeError("boom"), run_id=_run_id())

    def test_invocation_state_cleaned_up_after_chain_error(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        workflow_inv.span.is_recording.return_value = False
        handler.on_chain_error(error=RuntimeError("boom"), run_id=run_id)

        assert run_id not in handler._invocation_manager._invocations


# ---------------------------------------------------------------------------
# _find_nearest_agent
# ---------------------------------------------------------------------------


class TestFindNearestAgent:
    def test_returns_none_when_no_agent_in_ancestry(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        assert handler._find_nearest_agent(run_id) is None

    def test_finds_direct_parent_agent(self):
        handler, telemetry, _, agent_inv = _make_handler()
        parent_id = _run_id()
        child_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=parent_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        # Register the child as unclassified so it links to the parent
        handler._invocation_manager.add_invocation_state(
            child_id, parent_id, None
        )

        found = handler._find_nearest_agent(child_id)
        assert found is agent_inv

    def test_finds_grandparent_agent(self):
        handler, telemetry, _, agent_inv = _make_handler()
        grandparent_id = _run_id()
        parent_id = _run_id()
        child_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=grandparent_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )
        handler._invocation_manager.add_invocation_state(
            parent_id, grandparent_id, None
        )
        handler._invocation_manager.add_invocation_state(
            child_id, parent_id, None
        )

        found = handler._find_nearest_agent(child_id)
        assert found is agent_inv


# ---------------------------------------------------------------------------
# utils.make_input_message
# ---------------------------------------------------------------------------


class TestMakeInputMessage:
    def test_returns_empty_list_for_non_dict(self):
        assert make_input_message("not a dict") == []
        assert make_input_message(None) == []
        assert make_input_message(42) == []

    def test_empty_dict_returns_empty_list(self):
        assert make_input_message({}) == []

    def test_messages_key_with_human_message(self):
        msg = HumanMessage(content="Hello")
        result = make_input_message({"messages": [msg]})

        assert len(result) == 1
        assert isinstance(result[0], InputMessage)
        assert result[0].role == "user"
        assert len(result[0].parts) == 1
        assert isinstance(result[0].parts[0], Text)
        assert result[0].parts[0].content == "Hello"

    def test_messages_key_skips_empty_content(self):
        msg_empty = HumanMessage(content="")
        msg_valid = HumanMessage(content="Hi")
        result = make_input_message({"messages": [msg_empty, msg_valid]})

        assert len(result) == 1
        assert result[0].parts[0].content == "Hi"

    def test_messages_key_multiple_messages(self):
        msgs = [HumanMessage(content="First"), HumanMessage(content="Second")]
        result = make_input_message({"messages": msgs})

        assert len(result) == 2
        assert result[0].parts[0].content == "First"
        assert result[1].parts[0].content == "Second"

    def test_messages_key_takes_priority_over_other_fields(self):
        msg = HumanMessage(content="hello")
        result = make_input_message(
            {"messages": [msg], "user_query": "should be ignored"}
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "hello"

    def test_fallback_serializes_non_message_state_fields(self):
        result = make_input_message({"user_query": "what is 2+2?"})

        assert len(result) == 1
        assert result[0].role == "user"
        # The content should be a JSON serialization of the dict
        assert "user_query" in result[0].parts[0].content
        assert "what is 2+2?" in result[0].parts[0].content

    def test_fallback_excludes_intermediate_steps_key(self):
        # messages key absent → fallback path runs; intermediate_steps excluded
        result = make_input_message(
            {
                "user_query": "hi",
                "intermediate_steps": [("tool", "result")],
            }
        )

        assert len(result) == 1
        content = result[0].parts[0].content
        assert "intermediate_steps" not in content
        assert "user_query" in content

    def test_messages_key_present_skips_fallback_even_with_other_fields(self):
        # messages key present (even empty list) → early return, fallback not reached
        result = make_input_message(
            {
                "user_query": "ignored",
                "messages": [],
                "intermediate_steps": [("tool", "result")],
            }
        )

        assert result == []

    def test_fallback_returns_empty_when_all_values_are_none(self):
        result = make_input_message({"user_query": None, "context": None})
        assert result == []

    def test_fallback_returns_empty_when_only_excluded_keys(self):
        result = make_input_message(
            {"messages": None, "intermediate_steps": None}
        )
        assert result == []

    def test_messages_key_with_empty_list(self):
        # messages key present but empty → return empty list (no fallback)
        result = make_input_message({"messages": [], "user_query": "ignored"})
        assert result == []


# ---------------------------------------------------------------------------
# utils.make_output_message / make_last_output_message
# ---------------------------------------------------------------------------


class TestMakeOutputMessage:
    def test_returns_empty_list_for_non_dict(self):
        assert make_output_message("not a dict") == []
        assert make_output_message(None) == []

    def test_returns_empty_list_when_no_messages_key(self):
        assert make_output_message({"output": "hi"}) == []

    def test_returns_empty_list_when_messages_is_none(self):
        assert make_output_message({"messages": None}) == []

    def test_ai_message_produces_assistant_output(self):
        ai_msg = AIMessage(content="The answer is 42")
        result = make_output_message({"messages": [ai_msg]})

        assert len(result) == 1
        assert isinstance(result[0], OutputMessage)
        assert result[0].role == "assistant"
        assert result[0].finish_reason == "stop"
        assert result[0].parts[0].content == "The answer is 42"

    def test_non_ai_message_skipped(self):
        human_msg = HumanMessage(content="Hello")
        result = make_output_message({"messages": [human_msg]})
        assert result == []

    def test_ai_message_with_empty_content_skipped(self):
        ai_msg = AIMessage(content="")
        result = make_output_message({"messages": [ai_msg]})
        assert result == []

    def test_multiple_ai_messages_all_returned(self):
        msgs = [
            AIMessage(content="First response"),
            AIMessage(content="Second response"),
        ]
        result = make_output_message({"messages": msgs})

        assert len(result) == 2
        assert result[0].parts[0].content == "First response"
        assert result[1].parts[0].content == "Second response"

    def test_mixed_messages_only_ai_returned(self):
        msgs = [
            HumanMessage(content="question"),
            AIMessage(content="answer"),
            HumanMessage(content="follow-up"),
        ]
        result = make_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "answer"


class TestMakeLastOutputMessage:
    def test_returns_only_last_ai_message(self):
        msgs = [
            AIMessage(content="intermediate"),
            AIMessage(content="final answer"),
        ]
        result = make_last_output_message({"messages": msgs})

        assert len(result) == 1
        assert result[0].parts[0].content == "final answer"

    def test_returns_empty_when_no_ai_messages(self):
        result = make_last_output_message(
            {"messages": [HumanMessage(content="hi")]}
        )
        assert result == []

    def test_returns_empty_for_empty_outputs(self):
        assert make_last_output_message({}) == []

    def test_single_ai_message_returned(self):
        ai_msg = AIMessage(content="only response")
        result = make_last_output_message({"messages": [ai_msg]})

        assert len(result) == 1
        assert result[0].parts[0].content == "only response"


# ---------------------------------------------------------------------------
# utils.serialize
# ---------------------------------------------------------------------------


class TestSerialize:
    def test_none_returns_none(self):
        assert serialize(None) is None

    def test_dict_serialized_to_json(self):
        result = serialize({"key": "value"})
        assert result == '{"key": "value"}'

    def test_list_serialized_to_json(self):
        result = serialize([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_non_serializable_falls_back_to_str(self):
        class Custom:
            def __str__(self):
                return "custom_repr"

        result = serialize({"obj": Custom()})
        assert result is not None
        assert "custom_repr" in result

    def test_string_value(self):
        result = serialize("hello")
        assert result == '"hello"'


# ---------------------------------------------------------------------------
# input_messages / output_messages set on invocations via callback handler
# ---------------------------------------------------------------------------


class TestInputMessagesOnInvocations:
    def test_workflow_input_messages_set_from_messages_key(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()
        msg = HumanMessage(content="What is the weather?")

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"messages": [msg]},
            run_id=run_id,
            parent_run_id=None,
        )

        assigned = workflow_inv.input_messages
        assert len(assigned) == 1
        assert assigned[0].role == "user"
        assert assigned[0].parts[0].content == "What is the weather?"

    def test_workflow_input_messages_set_from_state_fallback(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={"user_query": "plan a trip"},
            run_id=run_id,
            parent_run_id=None,
        )

        assigned = workflow_inv.input_messages
        assert len(assigned) == 1
        assert "user_query" in assigned[0].parts[0].content
        assert "plan a trip" in assigned[0].parts[0].content

    def test_workflow_input_messages_empty_for_empty_inputs(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        assert workflow_inv.input_messages == []

    def test_agent_input_messages_set_from_messages_key(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()
        msg = HumanMessage(content="Solve x+2=5")

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={"messages": [msg]},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        assigned = agent_inv.input_messages
        assert len(assigned) == 1
        assert assigned[0].parts[0].content == "Solve x+2=5"

    def test_agent_input_messages_set_from_state_fallback(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={"problem": "integrate x^2"},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        assigned = agent_inv.input_messages
        assert len(assigned) == 1
        assert "integrate x^2" in assigned[0].parts[0].content


class TestOutputMessagesOnInvocations:
    def test_workflow_output_messages_set_on_chain_end(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        ai_msg = AIMessage(content="The final answer is 42")
        handler.on_chain_end(
            outputs={"messages": [ai_msg]},
            run_id=run_id,
        )

        assigned = workflow_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].role == "assistant"
        assert assigned[0].parts[0].content == "The final answer is 42"

    def test_workflow_output_messages_only_last_ai_message(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        msgs = [
            AIMessage(content="intermediate tool call"),
            AIMessage(content="final answer"),
        ]
        handler.on_chain_end(outputs={"messages": msgs}, run_id=run_id)

        assigned = workflow_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].parts[0].content == "final answer"

    def test_workflow_output_messages_empty_when_no_ai_messages(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_end(
            outputs={"messages": [HumanMessage(content="hi")]},
            run_id=run_id,
        )

        assert workflow_inv.output_messages == []

    def test_workflow_output_messages_empty_for_empty_outputs(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "LangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
        )

        handler.on_chain_end(outputs={}, run_id=run_id)

        assert workflow_inv.output_messages == []

    def test_agent_output_messages_set_on_chain_end(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        ai_msg = AIMessage(content="x = 3")
        handler.on_chain_end(outputs={"messages": [ai_msg]}, run_id=run_id)

        assigned = agent_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].parts[0].content == "x = 3"

    def test_agent_output_messages_only_last_ai_message(self):
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent"},
        )

        msgs = [
            AIMessage(content="let me think..."),
            AIMessage(content="the answer is 7"),
        ]
        handler.on_chain_end(outputs={"messages": msgs}, run_id=run_id)

        assigned = agent_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].parts[0].content == "the answer is 7"
