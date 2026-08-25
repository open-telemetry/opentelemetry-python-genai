# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Inheritance of gen_ai.conversation.id down a LangChain invocation tree.

The application passes a conversation id in the metadata of a single callback,
so every descendant of that run has to inherit it. These tests drive the
callbacks to build real multi-span trees and assert the attribute on each
exported span; key-precedence and single-callback resolution are covered at
the unit level in ``test_callback_handler.py`` and ``test_invocation_manager.py``.

semconv defines ``gen_ai.conversation.id`` on chat, invoke_agent, and
invoke_workflow spans only, so execute_tool and retrieval spans must not carry
it even while their descendants do.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from opentelemetry.instrumentation.genai.langchain.callback_handler import (
    OpenTelemetryLangChainCallbackHandler,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.handler import TelemetryHandler

_CONVERSATION_ID = GenAI.GEN_AI_CONVERSATION_ID

_OPENAI_SERIALIZED: dict[str, Any] = {"name": "ChatOpenAI"}
_OPENAI_INVOCATION_PARAMS: dict[str, Any] = {"model_name": "gpt-4"}
_LANGGRAPH_SERIALIZED: dict[str, Any] = {
    "name": "LangGraph",
    "id": ["langgraph"],
}


def _make_handler() -> tuple[
    OpenTelemetryLangChainCallbackHandler, InMemorySpanExporter
]:
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return (
        OpenTelemetryLangChainCallbackHandler(
            TelemetryHandler(tracer_provider=tracer_provider)
        ),
        span_exporter,
    )


def _spans_by_operation(span_exporter: InMemorySpanExporter) -> dict[str, Any]:
    """Map gen_ai.operation.name to the span carrying it."""
    return {
        span.attributes[GenAI.GEN_AI_OPERATION_NAME]: span
        for span in span_exporter.get_finished_spans()
        if span.attributes and GenAI.GEN_AI_OPERATION_NAME in span.attributes
    }


def _run_chat(
    handler: OpenTelemetryLangChainCallbackHandler,
    *,
    run_id: Any,
    parent_run_id: Any = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Drive a complete chat model call through the callbacks."""
    handler.on_chat_model_start(
        serialized=_OPENAI_SERIALIZED,
        messages=[[HumanMessage(content="What is 3 * 4?")]],
        run_id=run_id,
        parent_run_id=parent_run_id,
        metadata={"ls_provider": "openai", **(metadata or {})},
        invocation_params=_OPENAI_INVOCATION_PARAMS,
    )
    message = AIMessage(content="12")
    generation = ChatGeneration(message=message, text="12")
    generation.generation_info = {"finish_reason": "stop"}
    handler.on_llm_end(
        response=LLMResult(generations=[[generation]]),
        run_id=run_id,
        parent_run_id=parent_run_id,
    )


class TestInheritanceFromWorkflow:
    def test_chat_under_workflow_inherits_id(self):
        handler, span_exporter = _make_handler()
        workflow_run_id, chat_run_id = uuid4(), uuid4()

        handler.on_chain_start(
            serialized=_LANGGRAPH_SERIALIZED,
            inputs={},
            run_id=workflow_run_id,
            parent_run_id=None,
            metadata={"thread_id": "t1"},
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=workflow_run_id)
        handler.on_chain_end(outputs={}, run_id=workflow_run_id)

        spans = _spans_by_operation(span_exporter)
        assert spans["invoke_workflow"].attributes[_CONVERSATION_ID] == "t1"
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"


class TestInheritanceFromAgent:
    def test_chat_under_agent_inherits_id(self):
        handler, span_exporter = _make_handler()
        agent_run_id, chat_run_id = uuid4(), uuid4()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=agent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "thread_id": "t1"},
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=agent_run_id)
        handler.on_chain_end(outputs={}, run_id=agent_run_id)

        spans = _spans_by_operation(span_exporter)
        assert spans["invoke_agent"].attributes[_CONVERSATION_ID] == "t1"
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"

    def test_chat_under_tool_inherits_id_but_tool_does_not_carry_it(self):
        """A tool's descendants inherit the id; the tool span omits it."""
        handler, span_exporter = _make_handler()
        agent_run_id, tool_run_id, chat_run_id = uuid4(), uuid4(), uuid4()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=agent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "thread_id": "t1"},
        )
        handler.on_tool_start(
            serialized={"name": "multiply"},
            input_str="",
            run_id=tool_run_id,
            parent_run_id=agent_run_id,
            inputs={"a": 3, "b": 4},
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=tool_run_id)
        handler.on_tool_end(
            output=AIMessage(content="12"),
            run_id=tool_run_id,
            parent_run_id=agent_run_id,
        )
        handler.on_chain_end(outputs={}, run_id=agent_run_id)

        spans = _spans_by_operation(span_exporter)
        assert _CONVERSATION_ID not in spans["execute_tool"].attributes
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"

    def test_chat_under_retrieval_inherits_id_but_retrieval_does_not(self):
        handler, span_exporter = _make_handler()
        agent_run_id, retriever_run_id, chat_run_id = (
            uuid4(),
            uuid4(),
            uuid4(),
        )

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=agent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "thread_id": "t1"},
        )
        handler.on_retriever_start(
            serialized={"name": "my_retriever"},
            query="what is 3 * 4",
            run_id=retriever_run_id,
            parent_run_id=agent_run_id,
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=retriever_run_id)
        handler.on_retriever_end(
            documents=[Document(page_content="12")],
            run_id=retriever_run_id,
            parent_run_id=agent_run_id,
        )
        handler.on_chain_end(outputs={}, run_id=agent_run_id)

        spans = _spans_by_operation(span_exporter)
        assert _CONVERSATION_ID not in spans["retrieval"].attributes
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"

    def test_id_crosses_stacked_non_emitting_parents(self):
        """Inheritance survives a run of parents that never carry the attribute.

        A tool span omits gen_ai.conversation.id by semconv and an
        unclassified chain gets no span at all, so neither can be read back
        from. The id has to come from the invocation manager's bookkeeping.
        """
        handler, span_exporter = _make_handler()
        agent_run_id, tool_run_id, chain_run_id, chat_run_id = (
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
        )

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=agent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "thread_id": "t1"},
        )
        handler.on_tool_start(
            serialized={"name": "multiply"},
            input_str="",
            run_id=tool_run_id,
            parent_run_id=agent_run_id,
            inputs={"a": 3, "b": 4},
        )
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={},
            run_id=chain_run_id,
            parent_run_id=tool_run_id,
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=chain_run_id)
        handler.on_chain_end(outputs={}, run_id=chain_run_id)
        handler.on_tool_end(
            output=AIMessage(content="12"),
            run_id=tool_run_id,
            parent_run_id=agent_run_id,
        )
        handler.on_chain_end(outputs={}, run_id=agent_run_id)

        spans = _spans_by_operation(span_exporter)
        assert _CONVERSATION_ID not in spans["execute_tool"].attributes
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"

    def test_id_crosses_an_unclassified_chain(self):
        """Chains that emit no span still have to pass the id down."""
        handler, span_exporter = _make_handler()
        agent_run_id, chain_run_id, chat_run_id = uuid4(), uuid4(), uuid4()

        handler.on_chain_start(
            serialized={"name": "math_agent"},
            inputs={},
            run_id=agent_run_id,
            parent_run_id=None,
            metadata={"agent_name": "math_agent", "thread_id": "t1"},
        )
        # A nested chain with no agent or workflow signal is not given a span.
        handler.on_chain_start(
            serialized={"name": "RunnableSequence"},
            inputs={},
            run_id=chain_run_id,
            parent_run_id=agent_run_id,
        )
        _run_chat(handler, run_id=chat_run_id, parent_run_id=chain_run_id)
        handler.on_chain_end(outputs={}, run_id=chain_run_id)
        handler.on_chain_end(outputs={}, run_id=agent_run_id)

        spans = _spans_by_operation(span_exporter)
        assert spans["chat"].attributes[_CONVERSATION_ID] == "t1"

    def test_child_metadata_overrides_the_inherited_id(self):
        """Each run resolves its own id; a child is not pinned to the root's."""
        handler, span_exporter = _make_handler()
        workflow_run_id, chat_run_id = uuid4(), uuid4()

        handler.on_chain_start(
            serialized=_LANGGRAPH_SERIALIZED,
            inputs={},
            run_id=workflow_run_id,
            parent_run_id=None,
            metadata={"thread_id": "outer"},
        )
        _run_chat(
            handler,
            run_id=chat_run_id,
            parent_run_id=workflow_run_id,
            metadata={"thread_id": "inner"},
        )
        handler.on_chain_end(outputs={}, run_id=workflow_run_id)

        spans = _spans_by_operation(span_exporter)
        assert spans["invoke_workflow"].attributes[_CONVERSATION_ID] == "outer"
        assert spans["chat"].attributes[_CONVERSATION_ID] == "inner"
