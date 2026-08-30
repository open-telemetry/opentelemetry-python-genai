# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.llms import ChatMessage, MockFunctionCallingLLM
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import RetrieverTool

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.trace import SpanKind, StatusCode


class _StaticRetriever(BaseRetriever):
    def __init__(
        self,
        nodes: list[NodeWithScore] | None = None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__()
        self._nodes = nodes or []
        self._error = error
        self.similarity_top_k = 2

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        if self._error is not None:
            raise self._error
        return self._nodes


def _nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(id_="doc-1", text="Paris is in France."),
            score=0.95,
        ),
        NodeWithScore(
            node=TextNode(id_="doc-2", text="Berlin is in Germany."),
            score=None,
        ),
    ]


def _retrieval_span(span_exporter):
    spans = [
        span
        for span in span_exporter.get_finished_spans()
        if span.name == "retrieval"
    ]
    assert len(spans) == 1
    return spans[0]


def _assert_retrieval_content(span) -> None:
    attrs = dict(span.attributes or {})
    assert attrs[GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT] == (
        "Where are Paris and Berlin?"
    )
    documents = json.loads(
        attrs[GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS]
    )
    assert documents == [
        {
            "id": "doc-1",
            "content": "Paris is in France.",
            "score": 0.95,
        },
        {"id": "doc-2", "content": "Berlin is in Germany."},
    ]


def test_sync_retrieval_span_captures_query_and_documents(
    span_exporter, instrument_llama_index_with_content
) -> None:
    retriever = _StaticRetriever(_nodes())
    query = QueryBundle("Where are Paris and Berlin?")

    result = retriever.retrieve(query)

    assert result == _nodes()
    span = _retrieval_span(span_exporter)
    attrs = dict(span.attributes or {})
    assert span.kind == SpanKind.CLIENT
    assert attrs[GenAIAttributes.GEN_AI_OPERATION_NAME] == "retrieval"
    assert type(attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_K]) is float
    assert attrs[GenAIAttributes.GEN_AI_REQUEST_TOP_K] == 2.0
    _assert_retrieval_content(span)


@pytest.mark.asyncio
async def test_async_retrieval_span_captures_query_and_documents(
    span_exporter, instrument_llama_index_with_content
) -> None:
    retriever = _StaticRetriever(_nodes())

    result = await retriever.aretrieve("Where are Paris and Berlin?")

    assert result == _nodes()
    _assert_retrieval_content(_retrieval_span(span_exporter))


def test_retrieval_omits_content_by_default(
    span_exporter, instrument_llama_index
) -> None:
    _StaticRetriever(_nodes()).retrieve("Where are Paris and Berlin?")

    attrs = dict(_retrieval_span(span_exporter).attributes or {})
    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in attrs
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in attrs


def test_sync_retrieval_error_is_unchanged(
    span_exporter, instrument_llama_index
) -> None:
    error = RuntimeError("sync retrieval failed")

    with pytest.raises(RuntimeError) as caught:
        _StaticRetriever(error=error).retrieve("query")

    assert caught.value is error
    span = _retrieval_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "RuntimeError"


@pytest.mark.asyncio
async def test_async_retrieval_error_is_unchanged(
    span_exporter, instrument_llama_index
) -> None:
    error = ValueError("async retrieval failed")

    with pytest.raises(ValueError) as caught:
        await _StaticRetriever(error=error).aretrieve("query")

    assert caught.value is error
    span = _retrieval_span(span_exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[ErrorAttributes.ERROR_TYPE] == "ValueError"


@pytest.mark.asyncio
async def test_retrieval_span_is_nested_under_agent_tool(
    span_exporter, instrument_llama_index
) -> None:
    def response_generator(messages, **kwargs):
        if any(message.role.value == "tool" for message in messages):
            return ChatMessage(role="assistant", content="Paris is in France.")
        return ChatMessage(
            role="assistant",
            blocks=[
                ToolCallBlock(
                    tool_call_id="search-call",
                    tool_name="knowledge_search",
                    tool_kwargs={"query": "Where is Paris?"},
                )
            ],
        )

    agent = FunctionAgent(
        name="retrieval-agent",
        llm=MockFunctionCallingLLM(
            is_chat_model=True,
            response_generator=response_generator,
        ),
        tools=[
            RetrieverTool.from_defaults(
                _StaticRetriever(_nodes()),
                name="knowledge_search",
                description="Search the knowledge base.",
            )
        ],
        streaming=False,
    )

    result = await agent.run(user_msg="Where is Paris?")
    assert result.response.content == "Paris is in France."

    spans_by_name = {
        span.name: span for span in span_exporter.get_finished_spans()
    }
    agent_span = spans_by_name["invoke_agent retrieval-agent"]
    tool_span = spans_by_name["execute_tool knowledge_search"]
    retrieval_span = spans_by_name["retrieval"]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert retrieval_span.parent is not None
    assert retrieval_span.parent.span_id == tool_span.context.span_id
    assert retrieval_span.context.trace_id == agent_span.context.trace_id
