# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for on_chain_start / on_chain_end / on_chain_error in
OpenTelemetryLangChainCallbackHandler.

All TelemetryHandler interactions are mocked so that these tests exercise only
the callback-handler logic and the invocation-manager bookkeeping.
"""

import base64
import uuid
from unittest import mock

import pytest
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    RemoveMessage,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    LLMResult,
)

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.instrumentation.genai.langchain.utils import (
    _legacy_function_call_request,
    _media_part,
    _normalize_role,
    extract_token_details,
    make_input_message,
    make_last_output_message,
    make_output_message,
    serialize,
    to_input_messages,
    to_output_messages,
)
from opentelemetry.util.genai.invocation import (
    AgentInvocation,
    InferenceInvocation,
    RetrievalInvocation,
    WorkflowInvocation,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    InputMessage,
    OutputMessage,
    TextPart,
    ToolCallRequestPart,
    UriPart,
)

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


def _make_retrieval_inv_mock() -> mock.MagicMock:
    retrieval_inv = mock.MagicMock(spec=RetrievalInvocation)
    retrieval_inv.span = mock.MagicMock()
    retrieval_inv.span.is_recording.return_value = False
    retrieval_inv.query_text = None
    retrieval_inv.documents = None
    return retrieval_inv


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


def _make_handler_with_retrieval():
    """Like _make_handler but also wires up a retrieval mock."""
    handler, telemetry, workflow_inv, agent_inv = _make_handler()
    retrieval_inv = _make_retrieval_inv_mock()
    telemetry.retrieval.return_value = retrieval_inv
    return handler, telemetry, retrieval_inv


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

    def test_workflow_conversation_id_from_metadata(self):
        handler, _, workflow_inv, _ = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "MyLangGraph"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={"thread_id": "t1"},
        )

        assert workflow_inv.conversation_id == "t1"

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

    def test_conversation_id_prefers_session_id_over_conversation_id(self):
        """thread_id > session_id > conversation_id is the resolution order."""
        handler, _, _, agent_inv = _make_handler()
        run_id = _run_id()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=run_id,
            parent_run_id=None,
            metadata={
                "agent_name": "math_agent",
                "conversation_id": "c1",
                "session_id": "s1",
            },
        )

        assert agent_inv.conversation_id == "s1"

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


class TestOnChatModelStartConversationId:
    def test_conversation_id_from_metadata(self):
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="What is 3 * 4?")]],
            run_id=run_id,
            parent_run_id=None,
            metadata={"ls_provider": "openai", "thread_id": "t1"},
            invocation_params={"model_name": "gpt-4"},
        )

        assert telemetry.inference.return_value.conversation_id == "t1"

    def test_no_conversation_id_available(self):
        handler, telemetry, _, _ = _make_handler()
        run_id = _run_id()

        handler.on_chat_model_start(
            serialized={"name": "ChatOpenAI"},
            messages=[[HumanMessage(content="What is 3 * 4?")]],
            run_id=run_id,
            parent_run_id=None,
            metadata={"ls_provider": "openai"},
            invocation_params={"model_name": "gpt-4"},
        )

        assert telemetry.inference.return_value.conversation_id is None


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
        assert isinstance(result[0].parts[0], TextPart)
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

    def test_messages_key_converts_raw_role_content_tuples(self):
        result = make_input_message(
            {"messages": [("human", "Hello"), HumanMessage(content="Hi")]}
        )

        assert len(result) == 2
        assert result[0].role == "user"
        assert result[0].parts[0].content == "Hello"
        assert result[1].parts[0].content == "Hi"

    def test_messages_key_converts_role_content_dicts(self):
        result = make_input_message(
            {"messages": [{"role": "user", "content": "Hello"}]}
        )

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].parts[0].content == "Hello"

    @pytest.mark.parametrize(
        "bad_entry",
        [
            ("human",),  # too few items for (role, content)
            ("human", "hi", "extra"),  # too many items
            ("not_a_real_role", "hi"),  # unknown role string
            {"content": "no role key"},  # dict missing role
            {"role": "user"},  # dict missing content
            None,  # not a message at all
            42,  # not a message at all
            object(),  # arbitrary object
        ],
    )
    def test_messages_key_malformed_entry_does_not_raise(self, bad_entry):
        result = make_input_message(
            {"messages": [bad_entry, HumanMessage(content="Hi")]}
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "Hi"

    def test_messages_key_all_malformed_returns_empty_without_raising(self):
        result = make_input_message(
            {"messages": [("human",), None, {"content": "x"}]}
        )

        assert result == []

    def test_to_input_messages_never_raises_on_mixed_bad_input(self):
        result = to_input_messages(
            [object(), ("bad", "role", "shape"), HumanMessage(content="ok")]
        )

        assert len(result) == 1
        assert result[0].parts[0].content == "ok"

    def test_to_input_messages_empty_iterable_returns_empty(self):
        assert to_input_messages([]) == []

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
        assert result[0].finish_reason == ""
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

    def test_finish_reason_default_is_empty_string(self):
        # Workflow/agent output spans must not fabricate a finish reason:
        # util-genai filters empty strings out of
        # ``gen_ai.response.finish_reasons`` so the rollup span stays silent
        # while the per-call inference spans report the real values.
        result = make_output_message({"messages": [AIMessage(content="hi")]})
        assert result[0].finish_reason == ""

    def test_to_output_messages_propagates_explicit_finish_reason(self):
        # The inference path passes the provider's finish_reason through
        # ``to_output_messages``; the converter must forward it onto every
        # ``OutputMessage`` so util-genai can aggregate it into
        # ``gen_ai.response.finish_reasons``.
        msgs = [
            AIMessage(content="first"),
            AIMessage(content="second"),
        ]
        result = to_output_messages(msgs, finish_reason="tool_calls")
        assert [m.finish_reason for m in result] == [
            "tool_calls",
            "tool_calls",
        ]

    def test_to_output_messages_skips_non_ai_when_finish_reason_set(self):
        # finish_reason should never bleed onto non-AI messages: those are
        # filtered out entirely on the output side.
        result = to_output_messages(
            [HumanMessage(content="q"), AIMessage(content="a")],
            finish_reason="length",
        )
        assert len(result) == 1
        assert result[0].role == "assistant"
        assert result[0].finish_reason == "length"


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


# ---------------------------------------------------------------------------
# on_llm_end – tool call finish reasons
# ---------------------------------------------------------------------------


def _make_llm_invocation_mock() -> mock.MagicMock:
    inv = mock.MagicMock(spec=InferenceInvocation)
    inv.span = mock.MagicMock()
    inv.span.is_recording.return_value = False
    return inv


def _make_handler_with_llm_invocation(
    run_id: uuid.UUID,
) -> tuple[
    OpenTelemetryLangChainCallbackHandler, mock.MagicMock, mock.MagicMock
]:
    """Return a handler with an InferenceInvocation pre-registered for run_id."""
    telemetry = mock.MagicMock()
    llm_inv = _make_llm_invocation_mock()
    telemetry.inference.return_value = llm_inv

    handler = OpenTelemetryLangChainCallbackHandler(telemetry)
    handler._invocation_manager.add_invocation_state(run_id, None, llm_inv)
    return handler, telemetry, llm_inv


class TestOnLlmEndToolCalls:
    def test_openai_tool_calls_finish_reason_produces_tool_call_request(self):
        """finish_reason='tool_calls' (OpenAI) must produce ToolCallRequestPart parts."""
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        tool_call = {
            "name": "get_weather",
            "id": "call_123",
            "args": {"location": "Paris"},
        }
        ai_msg = AIMessage(
            content="", tool_calls=[tool_call], response_metadata={}
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "tool_calls"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assigned: list[OutputMessage] = llm_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].finish_reason == "tool_calls"
        assert len(assigned[0].parts) == 1
        part = assigned[0].parts[0]
        assert isinstance(part, ToolCallRequestPart)
        assert part.name == "get_weather"
        assert part.id == "call_123"
        assert part.arguments == {"location": "Paris"}

    def test_bedrock_tool_use_finish_reason_produces_tool_call_request(self):
        """finish_reason='tool_use' (Bedrock/Anthropic) must produce ToolCallRequestPart parts."""
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        tool_call = {
            "name": "get_weather",
            "id": "tooluse_abc",
            "args": {"location": "London"},
        }
        ai_msg = AIMessage(
            content="",
            tool_calls=[tool_call],
            response_metadata={"stopReason": "tool_use"},
        )
        # Bedrock path: generation_info is None; stopReason comes from response_metadata
        gen = ChatGeneration(message=ai_msg, generation_info=None)
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assigned: list[OutputMessage] = llm_inv.output_messages
        assert len(assigned) == 1
        assert assigned[0].finish_reason == "tool_use"
        assert len(assigned[0].parts) == 1
        part = assigned[0].parts[0]
        assert isinstance(part, ToolCallRequestPart)
        assert part.name == "get_weather"
        assert part.id == "tooluse_abc"
        assert part.arguments == {"location": "London"}


# ---------------------------------------------------------------------------
# on_retriever_start / on_retriever_end / on_retriever_error
# ---------------------------------------------------------------------------


class TestOnRetrieverStart:
    def test_retrieval_span_created(self):
        handler, telemetry, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="what is AI?",
            run_id=run_id,
        )

        telemetry.retrieval.assert_called_once()
        assert (
            handler._invocation_manager.get_invocation(run_id) is retrieval_inv
        )

    def test_conversation_id_not_passed_to_invocation(self):
        """semconv does not define gen_ai.conversation.id for retrieval."""
        handler, telemetry, _ = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="what is AI?",
            run_id=run_id,
            metadata={"thread_id": "t1"},
        )

        assert "conversation_id" not in telemetry.retrieval.call_args.kwargs

    def test_query_text_set_on_invocation(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="semantic search query",
            run_id=run_id,
        )

        assert retrieval_inv.query_text == "semantic search query"

    def test_provider_passed_from_metadata(self):
        handler, telemetry, _ = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="q",
            run_id=run_id,
            metadata={"ls_vector_store_provider": "Chroma"},
        )

        telemetry.retrieval.assert_called_once_with(
            provider="Chroma", request_model=None
        )

    def test_provider_none_when_metadata_absent(self):
        handler, telemetry, _ = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="q",
            run_id=run_id,
        )

        telemetry.retrieval.assert_called_once_with(
            provider=None, request_model=None
        )

    def test_request_model_passed_from_ls_embedding_model(self):
        handler, telemetry, _ = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="q",
            run_id=run_id,
            metadata={
                "ls_vector_store_provider": "Chroma",
                "ls_embedding_model": "text-embedding-3-small",
            },
        )

        telemetry.retrieval.assert_called_once_with(
            provider="Chroma", request_model="text-embedding-3-small"
        )

    def test_request_model_none_when_ls_embedding_model_absent(self):
        handler, telemetry, _ = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="q",
            run_id=run_id,
            metadata={"ls_vector_store_provider": "Chroma"},
        )

        telemetry.retrieval.assert_called_once_with(
            provider="Chroma", request_model=None
        )

    def test_registered_in_invocation_manager(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(
            serialized={},
            query="q",
            run_id=run_id,
        )

        assert run_id in handler._invocation_manager._invocations
        assert (
            handler._invocation_manager.get_invocation(run_id) is retrieval_inv
        )


class TestOnRetrieverEnd:
    def test_invocation_stopped(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=[], run_id=run_id)

        retrieval_inv.stop.assert_called_once()

    def test_documents_set_from_page_content(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        docs = [
            Document(page_content="doc one", metadata={"source": "s1"}),
            Document(page_content="doc two", metadata={}),
        ]

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=docs, run_id=run_id)

        assigned = retrieval_inv.documents
        assert len(assigned) == 2
        assert assigned[0]["content"] == "doc one"
        assert "source" not in assigned[0]
        assert assigned[1]["content"] == "doc two"

    def test_document_id_included_when_present(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        doc = Document(page_content="text", id="doc-123", metadata={})

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=[doc], run_id=run_id)

        assert retrieval_inv.documents[0]["id"] == "doc-123"

    def test_document_id_none_when_absent(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        doc = Document(page_content="text", metadata={})

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=[doc], run_id=run_id)

        assert retrieval_inv.documents[0]["id"] is None

    def test_state_cleaned_up_after_end(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        retrieval_inv.span.is_recording.return_value = False
        handler.on_retriever_end(documents=[], run_id=run_id)

        assert run_id not in handler._invocation_manager._invocations

    def test_documents_not_set_when_content_disabled(self):
        handler, telemetry, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()
        telemetry.should_capture_content.return_value = False

        docs = [Document(page_content="secret", metadata={})]
        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=docs, run_id=run_id)

        assert retrieval_inv.documents is None

    def test_documents_set_when_content_enabled(self):
        handler, telemetry, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()
        telemetry.should_capture_content.return_value = True

        docs = [Document(page_content="visible", metadata={})]
        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        handler.on_retriever_end(documents=docs, run_id=run_id)

        assert retrieval_inv.documents[0]["content"] == "visible"

    def test_unknown_run_id_does_not_raise(self):
        handler, _, _ = _make_handler_with_retrieval()
        handler.on_retriever_end(documents=[], run_id=_run_id())


class TestOnRetrieverError:
    def test_invocation_failed(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        err = RuntimeError("retrieval failed")
        handler.on_retriever_error(error=err, run_id=run_id)

        retrieval_inv.fail.assert_called_once_with(err)

    def test_state_cleaned_up_after_error(self):
        handler, _, retrieval_inv = _make_handler_with_retrieval()
        run_id = _run_id()

        handler.on_retriever_start(serialized={}, query="q", run_id=run_id)
        retrieval_inv.span.is_recording.return_value = False
        handler.on_retriever_error(error=RuntimeError("boom"), run_id=run_id)

        assert run_id not in handler._invocation_manager._invocations

    def test_unknown_run_id_does_not_raise(self):
        handler, _, _ = _make_handler_with_retrieval()
        handler.on_retriever_error(
            error=RuntimeError("boom"), run_id=_run_id()
        )


# on_llm_end – token usage break-downs
# ---------------------------------------------------------------------------


class TestOnLlmEndTokenDetails:
    def test_cache_and_reasoning_tokens_set_on_invocation(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="hi there",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {
                    "cache_creation": 3,
                    "cache_read": 2,
                },
                "output_token_details": {"reasoning": 5},
            },
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.input_tokens == 10
        assert llm_inv.cache_creation_input_tokens == 3
        assert llm_inv.cache_read_input_tokens == 2
        assert llm_inv.thinking_tokens == 5
        assert llm_inv.output_tokens == 20

    def test_audio_tokens_ignored(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {"audio": 5},
                "output_token_details": {"audio": 4},
            },
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.input_tokens == 10
        assert llm_inv.output_tokens == 20


# ---------------------------------------------------------------------------
# utils.extract_token_details
# ---------------------------------------------------------------------------


def test_extract_token_details_cache_and_reasoning():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "input_token_details": {"cache_creation": 3, "cache_read": 2},
        "output_token_details": {"reasoning": 5},
    }
    details = extract_token_details(usage)
    assert details == {
        "cache_creation_input_tokens": 3,
        "cache_read_input_tokens": 2,
        "reasoning_tokens": 5,
    }


def test_extract_token_details_ignores_audio_tokens():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_token_details": {"audio": 5},
        "output_token_details": {"audio": 4},
    }
    assert extract_token_details(usage) == {}


def test_extract_token_details_zero_values_omitted():
    usage = {
        "input_tokens": 10,
        "output_tokens": 20,
        "input_token_details": {"cache_creation": 0, "cache_read": 0},
    }
    assert extract_token_details(usage) == {}


def test_extract_token_details_no_details_key():
    assert extract_token_details({"input_tokens": 1, "output_tokens": 2}) == {}

    def test_legacy_function_call_finish_reason_produces_tool_call_request(
        self,
    ):
        """Pre-tools OpenAI ``function_call`` must surface as a ToolCallRequestPart."""
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="",
            additional_kwargs={
                "function_call": {
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                }
            },
        )
        gen = ChatGeneration(
            message=ai_msg,
            generation_info={"finish_reason": "function_call"},
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assigned: list[OutputMessage] = llm_inv.output_messages
        assert len(assigned) == 1
        assert len(assigned[0].parts) == 1
        part = assigned[0].parts[0]
        assert isinstance(part, ToolCallRequestPart)
        assert part.name == "get_weather"
        assert part.arguments == {"city": "Paris"}


# ---------------------------------------------------------------------------
# utils - legacy OpenAI function_call (_legacy_function_call_request)
# ---------------------------------------------------------------------------


def test_legacy_function_call_dict_arguments():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "get_weather",
                "arguments": {"city": "New York"},
            }
        },
    )
    call = _legacy_function_call_request(message)
    assert isinstance(call, ToolCallRequestPart)
    assert call.name == "get_weather"
    assert call.arguments == {"city": "New York"}


def test_legacy_function_call_string_arguments_parsed():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "get_weather",
                "arguments": '{"city": "New York"}',
            }
        },
    )
    call = _legacy_function_call_request(message)
    assert isinstance(call, ToolCallRequestPart)
    assert call.arguments == {"city": "New York"}


def test_legacy_function_call_absent_returns_none():
    assert _legacy_function_call_request(AIMessage(content="hi")) is None


def test_to_input_messages_includes_legacy_function_call():
    message = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {"name": "f", "arguments": {"x": 1}},
        },
    )
    messages = to_input_messages([message])
    assert len(messages) == 1
    assert any(
        isinstance(p, ToolCallRequestPart) and p.name == "f"
        for p in messages[0].parts
    )


# ---------------------------------------------------------------------------
# on_llm_end – response model resolution (RAPI header + llm_output)
# ---------------------------------------------------------------------------


class TestOnLlmEndResponseModel:
    def test_served_model_from_rapi_header(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="hello",
            response_metadata={
                "headers": {"x-ms-served-model": "gpt-4.1-2025-04-14"}
            },
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name == "gpt-4.1-2025-04-14"

    def test_served_model_header_case_insensitive(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="hello",
            response_metadata={
                "headers": {"X-MS-Served-Model": "gpt-4.1-2025-04-14"}
            },
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name == "gpt-4.1-2025-04-14"

    def test_served_model_overrides_llm_output_model_name(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(
            content="hello",
            response_metadata={
                "headers": {"x-ms-served-model": "served-from-header"}
            },
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(
            generations=[[gen]],
            llm_output={"model_name": "gpt-4.1-body", "id": "resp_123"},
        )

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name == "served-from-header"
        assert llm_inv.response_id == "resp_123"

    def test_llm_output_model_fallback(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        ai_msg = AIMessage(content="hello", response_metadata={})
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(
            generations=[[gen]], llm_output={"model": "gpt-4.1-fallback"}
        )

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name == "gpt-4.1-fallback"

    def test_unsupported_header_ignored(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)
        llm_inv.response_model_name = None

        ai_msg = AIMessage(
            content="hello",
            response_metadata={"headers": {"x-request-id": "abc-123"}},
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name is None

    def test_empty_header_value_ignored(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)
        llm_inv.response_model_name = "gpt-4o"

        ai_msg = AIMessage(
            content="hello",
            response_metadata={"headers": {"x-ms-served-model": " "}},
        )
        gen = ChatGeneration(
            message=ai_msg, generation_info={"finish_reason": "stop"}
        )
        response = LLMResult(generations=[[gen]])

        handler.on_llm_end(response=response, run_id=run_id)

        assert llm_inv.response_model_name == "gpt-4o"


# ---------------------------------------------------------------------------
# on_llm_end – streamed output messages
# ---------------------------------------------------------------------------


def test_streamed_output_message_role_is_assistant():
    run_id = _run_id()
    handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

    gen = ChatGenerationChunk(message=AIMessageChunk(content="hello"))

    handler.on_llm_end(response=LLMResult(generations=[[gen]]), run_id=run_id)

    (output_message,) = llm_inv.output_messages
    assert output_message.role == "assistant"


# ---------------------------------------------------------------------------
# utils._normalize_role
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_role",
    [
        (AIMessage(content="hi"), "assistant"),
        (AIMessageChunk(content="hi"), "assistant"),
        (HumanMessage(content="hi"), "user"),
        (HumanMessageChunk(content="hi"), "user"),
        (SystemMessage(content="hi"), "system"),
        (SystemMessageChunk(content="hi"), "system"),
        (ToolMessage(content="hi", tool_call_id="call_1"), "tool"),
        (ToolMessageChunk(content="hi", tool_call_id="call_1"), "tool"),
        (FunctionMessage(content="hi", name="f"), "tool"),
        (FunctionMessageChunk(content="hi", name="f"), "tool"),
    ],
)
def test_normalize_role_resolves_chunk_variants(message, expected_role):
    """Chunk classes report their class name as ``.type``
    (``AIMessageChunk.type == "AIMessageChunk"``), so they only resolve to a
    spec role when matched by class."""
    assert _normalize_role(message) == expected_role


@pytest.mark.parametrize(
    "message,expected_role",
    [
        (ChatMessage(content="hi", role="assistant"), "assistant"),
        (ChatMessage(content="hi", role="custom"), "custom"),
        (ChatMessageChunk(content="hi", role="custom"), "custom"),
    ],
)
def test_normalize_role_reads_the_chat_message_role(message, expected_role):
    """``ChatMessage`` keeps its speaker in ``role`` rather than in the class."""
    assert _normalize_role(message) == expected_role


def test_normalize_role_returns_none_for_unmapped_class():
    assert _normalize_role(RemoveMessage(id="abc")) is None


def test_chat_message_role_reaches_the_input_side():
    (message,) = to_input_messages([ChatMessage(content="hi", role="custom")])
    assert message.role == "custom"


# on_llm_end – streamed responses
#
# Streaming reaches on_llm_end with an LLMResult that LangChain assembles from
# the merged chunks alone, so ``llm_output`` — the usual source of
# ``gen_ai.response.model`` and ``gen_ai.response.id`` — is never populated.
# ---------------------------------------------------------------------------


class TestOnLlmEndStreamedResponse:
    def test_model_and_id_from_response_metadata(self):
        """Both fields are read off the message's ``response_metadata``.

        LangChain fills it with the union of ``generation_info`` and whatever
        the provider wrote onto the message, so this is the one place that
        covers every provider. The ``generation_info`` half of that union is
        covered end to end by
        ``test_chat_openai_streamed_response_model``, which drives a
        real ``.stream()`` so LangChain performs the copy.
        """
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        gen = ChatGenerationChunk(
            message=AIMessageChunk(
                content="hello",
                response_metadata={
                    "model_name": "claude-sonnet-4-5",
                    "id": "msg_01ABC",
                },
            )
        )

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_model_name == "claude-sonnet-4-5"
        assert llm_inv.response_id == "msg_01ABC"

    def test_model_from_generation_info(self):
        """A generation that skipped LangChain's merge still reports."""
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        gen = ChatGenerationChunk(
            message=AIMessageChunk(content="hello"),
            generation_info={"model_name": "claude-sonnet-4-5"},
        )

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_model_name == "claude-sonnet-4-5"

    def test_response_metadata_wins_over_generation_info(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        gen = ChatGenerationChunk(
            message=AIMessageChunk(
                content="hello",
                response_metadata={"model_name": "from-response-metadata"},
            ),
            generation_info={"model_name": "from-generation-info"},
        )

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_model_name == "from-response-metadata"

    def test_message_id_not_used_as_response_id(self):
        """LangChain puts its own run id on ``message.id`` when the provider
        supplies none, which must not surface as ``gen_ai.response.id``."""
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)
        llm_inv.response_id = None

        gen = ChatGenerationChunk(
            message=AIMessageChunk(
                content="hello", id=f"run--{run_id}", chunk_position="last"
            )
        )

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_id is None

    def test_llm_output_takes_precedence(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        gen = ChatGenerationChunk(
            message=AIMessageChunk(content="hello"),
            generation_info={"model_name": "from-generation"},
        )

        handler.on_llm_end(
            response=LLMResult(
                generations=[[gen]],
                llm_output={"model_name": "from-llm-output"},
            ),
            run_id=run_id,
        )

        assert llm_inv.response_model_name == "from-llm-output"

    def test_served_model_header_takes_precedence(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)

        gen = ChatGenerationChunk(
            message=AIMessageChunk(
                content="hello",
                response_metadata={
                    "headers": {"x-ms-served-model": "served-from-header"}
                },
            ),
            generation_info={"model_name": "from-generation"},
        )

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_model_name == "served-from-header"

    def test_no_model_reported_leaves_attribute_unset(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)
        llm_inv.response_model_name = None

        gen = ChatGenerationChunk(message=AIMessageChunk(content="hello"))

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_model_name is None

    def test_no_id_reported_leaves_attribute_unset(self):
        run_id = _run_id()
        handler, _, llm_inv = _make_handler_with_llm_invocation(run_id)
        llm_inv.response_id = None

        gen = ChatGenerationChunk(message=AIMessageChunk(content="hello"))

        handler.on_llm_end(
            response=LLMResult(generations=[[gen]]), run_id=run_id
        )

        assert llm_inv.response_id is None


# ---------------------------------------------------------------------------
# utils._media_part - LangChain multimodal image block parsing
# ---------------------------------------------------------------------------

_REAL_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00"
    b"\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc"
    b"\xf8\xcf\xc0\xf0\x1f\x00\x05\x05\x02\x00\xa1\r\xf7\xdf\x00\x00\x00"
    b"\x00IEND\xaeB`\x82"
)
_REAL_PNG_B64 = base64.b64encode(_REAL_PNG_BYTES).decode("ascii")


def test_media_part_openai_image_url_dict():
    item = {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/a.png"


def test_media_part_openai_image_url_string():
    item = {"type": "image_url", "image_url": "https://example.com/b.png"}
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/b.png"


def test_media_part_anthropic_base64_source_returns_blob():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "R0lGODlh",
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == b"GIF89a"


def test_media_part_anthropic_base64_source_without_media_type():
    item = {
        "type": "image",
        "source": {"type": "base64", "data": "QUJD"},
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type is None
    assert part.content == b"ABC"


def test_media_part_base64_source_decodes_real_png():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _REAL_PNG_B64,
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_url_source_returns_uri():
    item = {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/c.png"},
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/c.png"


def test_media_part_unrecognized_returns_none():
    assert _media_part({"type": "text", "text": "hi"}) is None
    assert _media_part({"type": "image_url", "image_url": {}}) is None
    assert (
        _media_part({"type": "image_url", "image_url": {"url": 123}}) is None
    )
    assert _media_part({"type": "image", "source": "nope"}) is None
    assert (
        _media_part({"type": "image", "source": {"type": "base64", "data": 5}})
        is None
    )
    assert (
        _media_part({"type": "image", "source": {"type": "url", "url": ""}})
        is None
    )


def test_media_part_malformed_base64_returns_none():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": "not!valid!base64!",
        },
    }
    assert _media_part(item) is None


def test_media_part_openai_real_png_data_uri_returns_blob():
    item = {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_real_png_source_returns_blob():
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": _REAL_PNG_B64,
        },
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_real_png_corrupted_base64_returns_none():
    corrupted = _REAL_PNG_B64[:10] + "@@@@" + _REAL_PNG_B64[14:]
    item = {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": corrupted,
        },
    }
    assert _media_part(item) is None


def test_media_part_real_png_url_source_returns_uri():
    item = {
        "type": "image",
        "source": {
            "type": "url",
            "url": "https://example.com/real-image.png",
        },
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/real-image.png"


def test_media_part_standard_v1_base64_block_returns_blob():
    item = {
        "type": "image",
        "base64": _REAL_PNG_B64,
        "mime_type": "image/png",
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v1_url_block_returns_uri():
    item = {"type": "image", "url": "https://example.com/a.png"}
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/a.png"


def test_media_part_standard_block_without_mime_type_returns_blob():
    part = _media_part({"type": "image", "base64": _REAL_PNG_B64})
    assert isinstance(part, BlobPart)
    assert part.mime_type is None
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v03_block_without_mime_type_returns_blob():
    part = _media_part(
        {"type": "image", "source_type": "base64", "data": _REAL_PNG_B64}
    )
    assert isinstance(part, BlobPart)
    assert part.mime_type is None


def test_media_part_non_dict_source_falls_through_to_standard_keys():
    """A non-dict ``source`` must not shadow a standard payload."""
    part = _media_part(
        {
            "type": "image",
            "source": "not-a-dict",
            "base64": _REAL_PNG_B64,
            "mime_type": "image/png",
        }
    )
    assert isinstance(part, BlobPart)
    assert part.content == _REAL_PNG_BYTES


def test_media_part_anthropic_source_takes_precedence_over_standard_keys():
    """Anthropic's ``source`` is checked before the standard top-level keys."""
    part = _media_part(
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/from.png"},
            "url": "https://example.com/ignored.png",
        }
    )
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/from.png"


def test_media_part_v03_url_source_type_does_not_fall_through_to_base64():
    """An explicit ``source_type`` pins the variant, even when it is broken."""
    part = _media_part(
        {
            "type": "image",
            "source_type": "url",
            "url": None,
            "base64": _REAL_PNG_B64,
        }
    )
    assert part is None


def test_media_part_standard_v03_base64_block_returns_blob():
    item = {
        "type": "image",
        "source_type": "base64",
        "data": _REAL_PNG_B64,
        "mime_type": "image/png",
    }
    part = _media_part(item)
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_standard_v03_url_block_returns_uri():
    item = {
        "type": "image",
        "source_type": "url",
        "url": "https://example.com/b.png",
    }
    part = _media_part(item)
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/b.png"


def test_media_part_standard_block_corrupted_base64_returns_none():
    corrupted = _REAL_PNG_B64[:10] + "@@@@" + _REAL_PNG_B64[14:]
    assert (
        _media_part(
            {"type": "image", "base64": corrupted, "mime_type": "image/png"}
        )
        is None
    )


def test_media_part_standard_block_without_payload_returns_none():
    assert _media_part({"type": "image", "mime_type": "image/png"}) is None


@pytest.mark.parametrize(
    "block,expected,expected_value",
    [
        (
            {
                "type": "image",
                "base64": _REAL_PNG_B64,
                "mime_type": "image/png",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {"type": "image", "url": "https://example.com/a.png"},
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {
                "type": "image",
                "source_type": "base64",
                "data": _REAL_PNG_B64,
                "mime_type": "image/png",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {
                "type": "image",
                "source_type": "url",
                "url": "https://e/b.png",
            },
            UriPart,
            "https://e/b.png",
        ),
        (
            {"type": "image", "source_type": "id", "id": "file-123"},
            FilePart,
            "file-123",
        ),
        (
            {"type": "image", "id": "file-456"},
            FilePart,
            "file-456",
        ),
        (
            {
                "type": "image",
                "source": {"type": "file", "file_id": "file-789"},
            },
            FilePart,
            "file-789",
        ),
    ],
)
def test_langchain_standard_image_blocks_are_captured(
    block, expected, expected_value
):
    messages = to_input_messages([HumanMessage(content=[block])])
    assert messages, "message dropped entirely - no parts extracted"
    part = next(p for p in messages[0].parts if isinstance(p, expected))
    assert part.modality == "image"
    assert _part_payload(part) == expected_value


@pytest.mark.parametrize(
    "block,expected,expected_value",
    [
        (
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
            },
            BlobPart,
            _REAL_PNG_BYTES,
        ),
        (
            {"type": "input_image", "image_url": "https://example.com/a.png"},
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {
                "type": "input_image",
                "image_url": {"url": "https://example.com/a.png"},
            },
            UriPart,
            "https://example.com/a.png",
        ),
        (
            {"type": "input_image", "file_id": "file-123"},
            FilePart,
            "file-123",
        ),
    ],
)
def test_responses_api_input_image_blocks_are_captured(
    block, expected, expected_value
):
    messages = to_input_messages([HumanMessage(content=[block])])
    assert messages, "message dropped entirely - no parts extracted"
    part = next(p for p in messages[0].parts if isinstance(p, expected))
    assert part.modality == "image"
    assert _part_payload(part) == expected_value


def _part_payload(part):
    if isinstance(part, BlobPart):
        return part.content
    if isinstance(part, UriPart):
        return part.uri
    return part.file_id


def test_media_part_responses_api_input_image_base64_url():
    part = _media_part(
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{_REAL_PNG_B64}",
        }
    )
    assert isinstance(part, BlobPart)
    assert part.mime_type == "image/png"
    assert part.modality == "image"
    assert part.content == _REAL_PNG_BYTES


def test_media_part_responses_api_input_image_file_id_returns_file_part():
    # Provider-hosted image: no bytes and no URL, but the reference is still
    # worth recording.
    part = _media_part({"type": "input_image", "file_id": "file-123"})
    assert isinstance(part, FilePart)
    assert part.file_id == "file-123"
    assert part.modality == "image"
    assert part.mime_type is None


def test_media_part_anthropic_file_source_returns_file_part():
    part = _media_part(
        {
            "type": "image",
            "source": {
                "type": "file",
                "file_id": "file-789",
                "media_type": "image/png",
            },
        }
    )
    assert isinstance(part, FilePart)
    assert part.file_id == "file-789"
    assert part.mime_type == "image/png"


def test_media_part_anthropic_file_source_without_file_id_returns_none():
    assert _media_part({"type": "image", "source": {"type": "file"}}) is None


def test_responses_api_input_text_block_is_captured():
    messages = to_input_messages(
        [
            HumanMessage(
                content=[{"type": "input_text", "text": "what is this?"}]
            )
        ]
    )
    assert len(messages) == 1
    parts = messages[0].parts
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "what is this?"


def test_responses_api_output_text_block_is_captured():
    # langchain-openai accepts "output_text" on an assistant turn fed back in
    # as request input.
    messages = to_input_messages(
        [AIMessage(content=[{"type": "output_text", "text": "the answer"}])]
    )
    assert len(messages) == 1
    assert messages[0].role == "assistant"
    parts = messages[0].parts
    assert len(parts) == 1
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "the answer"


def test_message_survives_unconvertible_image_block():
    # An image block with no bytes, url, or file id has nothing to record.
    messages = to_input_messages(
        [HumanMessage(content=[{"type": "image", "detail": "high"}])]
    )
    assert messages, "message dropped entirely"
    assert messages[0].role == "user"
    assert messages[0].parts == []


def test_text_kept_when_image_block_is_unconvertible():
    messages = to_input_messages(
        [
            HumanMessage(
                content=[
                    {"type": "input_text", "text": "what is in this image?"},
                    {"type": "input_image", "detail": "high"},
                ]
            )
        ]
    )
    assert messages, "message dropped entirely"
    assert any(isinstance(p, TextPart) for p in messages[0].parts)


def test_message_with_only_unknown_blocks_is_kept():
    messages = to_input_messages(
        [HumanMessage(content=[{"type": "input_file", "file_id": "file-1"}])]
    )
    assert messages
    assert messages[0].parts == []


def test_empty_message_is_still_dropped():
    assert to_input_messages([HumanMessage(content="")]) == []
    assert to_input_messages([HumanMessage(content=[])]) == []


@pytest.mark.parametrize("content", [[""], ["", ""], [{}], ["", {}]])
def test_message_of_blank_blocks_is_dropped(content):
    # Blank blocks are empty content, not content we failed to convert.
    assert to_input_messages([HumanMessage(content=content)]) == []


def test_output_message_survives_unconvertible_blocks():
    # A dropped output message would also lose its finish_reason.
    messages = to_output_messages(
        [AIMessage(content=[{"type": "input_file", "file_id": "file-1"}])],
        finish_reason="stop",
    )
    assert messages
    assert messages[0].parts == []
    assert messages[0].finish_reason == "stop"


def test_empty_output_message_is_still_dropped():
    assert to_output_messages([AIMessage(content="")]) == []


def test_to_input_messages_extracts_image_part():
    image_url = "data:image/jpeg;base64,QUJD"
    content = [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    messages = to_input_messages([HumanMessage(content=content)])
    assert len(messages) == 1
    parts = messages[0].parts

    assert any(isinstance(p, BlobPart) for p in parts)
    blob = next(p for p in parts if isinstance(p, BlobPart))
    assert blob.mime_type == "image/jpeg"
    assert blob.content == b"ABC"


def test_on_chat_model_start_skips_input_messages_when_content_disabled():
    """Gating happens in the callback handler, not inside the utils."""
    run_id = _run_id()
    handler, telemetry, llm_inv = _make_handler_with_llm_invocation(run_id)
    telemetry.should_capture_content.return_value = False

    content = [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
        },
    ]
    handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content=content)]],
        run_id=_run_id(),
        invocation_params={"model_name": "gpt-4o"},
    )

    assert llm_inv.input_messages == []


def test_on_chat_model_start_captures_input_messages_when_content_enabled():
    run_id = _run_id()
    handler, telemetry, llm_inv = _make_handler_with_llm_invocation(run_id)
    telemetry.should_capture_content.return_value = True

    content = [
        {"type": "text", "text": "What's in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_REAL_PNG_B64}"},
        },
    ]
    handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content=content)]],
        run_id=_run_id(),
        invocation_params={"model_name": "gpt-4o"},
    )

    parts = llm_inv.input_messages[0].parts
    assert any(isinstance(p, TextPart) for p in parts)
    blob = next(p for p in parts if isinstance(p, BlobPart))
    assert blob.content == _REAL_PNG_BYTES
