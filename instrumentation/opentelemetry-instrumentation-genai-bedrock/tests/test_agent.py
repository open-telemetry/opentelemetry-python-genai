# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Amazon Bedrock Agent Runtime (invoke_agent and retrieve)."""

from __future__ import annotations

import json
from typing import Any

import aiobotocore.session
import boto3
import pytest
import pytest_asyncio
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from opentelemetry.instrumentation.genai.bedrock.extractors import (
    extract_retrieve_response,
)
from opentelemetry.semconv._incubating.attributes import (
    error_attributes as ErrorAttributes,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.semconv.attributes import (
    server_attributes as ServerAttributes,
)
from opentelemetry.trace import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler


@pytest.fixture
def agent_client():
    return boto3.client("bedrock-agent-runtime", region_name="us-east-1")


@pytest_asyncio.fixture
async def async_agent_client():
    session = aiobotocore.session.get_session()
    async with session.create_client(
        "bedrock-agent-runtime", region_name="us-east-1"
    ) as client:
        yield client


class _MockAsyncEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> _MockAsyncEventStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class _FailingAsyncEventStream:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __aiter__(self) -> _FailingAsyncEventStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            item = next(self._iter)
            if "fail" in item:
                raise ConnectionError("Async stream disconnected")
            return item
        except StopIteration:
            raise StopAsyncIteration


# --- Sync invoke_agent tests ---


def test_invoke_agent_sync_with_content(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "session-123",
            "completion": [
                {"chunk": {"bytes": b"Hello "}},
                {"chunk": {"bytes": b"from agent!"}},
            ],
        },
        expected_params={
            "agentId": "agent-12345",
            "agentAliasId": "alias-12345",
            "sessionId": "session-123",
            "inputText": "Hi",
        },
    )

    with stubber:
        response = agent_client.invoke_agent(
            agentId="agent-12345",
            agentAliasId="alias-12345",
            sessionId="session-123",
            inputText="Hi",
        )
        chunks = list(response["completion"])
        assert len(chunks) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_ID) == "agent-12345"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_VERSION)
        == "alias-12345"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "session-123"
    )
    assert (
        span.attributes.get(ServerAttributes.SERVER_ADDRESS)
        == "bedrock-agent-runtime.us-east-1.amazonaws.com"
    )

    # Content capture enabled
    input_msgs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    )
    assert len(input_msgs) == 1
    assert input_msgs[0]["role"] == "user"
    assert input_msgs[0]["parts"][0]["content"] == "Hi"

    output_msgs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    )
    assert len(output_msgs) == 1
    assert output_msgs[0]["role"] == "assistant"
    assert output_msgs[0]["parts"][0]["content"] == "Hello from agent!"


def test_invoke_agent_sync_no_content(
    agent_client,
    instrument_no_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "session-456",
            "completion": [
                {"chunk": {"bytes": b"Secret answer"}},
            ],
        },
        expected_params={
            "agentId": "agent-12345",
            "agentAliasId": "alias-12345",
            "sessionId": "session-456",
            "inputText": "Secret question",
        },
    )

    with stubber:
        response = agent_client.invoke_agent(
            agentId="agent-12345",
            agentAliasId="alias-12345",
            sessionId="session-456",
            inputText="Secret question",
        )
        list(response["completion"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_ID) == "agent-12345"
    )


def test_invoke_agent_sync_error(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber.add_client_error(
        "invoke_agent",
        service_error_code="ResourceNotFoundException",
        service_message="Agent not found",
        expected_params={
            "agentId": "agent-missing",
            "agentAliasId": "alias-12345",
            "sessionId": "session-123",
            "inputText": "Hi",
        },
    )

    with stubber:
        with pytest.raises(ClientError):
            agent_client.invoke_agent(
                agentId="agent-missing",
                agentAliasId="alias-12345",
                sessionId="session-123",
                inputText="Hi",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) in (
        "ResourceNotFoundException",
        "botocore.errorfactory.ResourceNotFoundException",
    )


def test_invoke_agent_sync_stream_error(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    class FailingEventStream:
        def __iter__(self):
            yield {"chunk": {"bytes": b"part 1"}}
            raise ConnectionError("Stream broke")

    stubber = Stubber(agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "session-123",
            "completion": FailingEventStream(),
        },
        expected_params={
            "agentId": "agent-12345",
            "agentAliasId": "alias-12345",
            "sessionId": "session-123",
            "inputText": "Hi",
        },
    )

    with stubber:
        response = agent_client.invoke_agent(
            agentId="agent-12345",
            agentAliasId="alias-12345",
            sessionId="session-123",
            inputText="Hi",
        )
        with pytest.raises(ConnectionError):
            list(response["completion"])

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "ConnectionError"


def test_invoke_agent_sync_caller_error_during_stream(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "session-123",
            "completion": [
                {"chunk": {"bytes": b"part 1"}},
                {"chunk": {"bytes": b"part 2"}},
            ],
        },
        expected_params={
            "agentId": "agent-12345",
            "agentAliasId": "alias-12345",
            "sessionId": "session-123",
            "inputText": "Hi",
        },
    )

    with stubber:
        response = agent_client.invoke_agent(
            agentId="agent-12345",
            agentAliasId="alias-12345",
            sessionId="session-123",
            inputText="Hi",
        )
        with pytest.raises(RuntimeError):
            with response["completion"] as stream:
                for _ in stream:
                    raise RuntimeError("caller error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


# --- Sync retrieve tests ---


def test_retrieve_sync_with_content(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber.add_response(
        "retrieve",
        service_response={
            "retrievalResults": [
                {
                    "content": {"text": "Document 1 content"},
                    "location": {
                        "type": "S3",
                        "s3Location": {"uri": "s3://my-bucket/doc1.txt"},
                    },
                    "score": 0.95,
                    "metadata": {"author": "Alice"},
                },
                {
                    "content": {"text": "Document 2 content"},
                    "location": {
                        "type": "WEB",
                        "webLocation": {"url": "https://example.com/doc2"},
                    },
                    "score": 0.85,
                },
            ],
        },
        expected_params={
            "knowledgeBaseId": "kb-12345678",
            "retrievalQuery": {"text": "What is OpenTelemetry?"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": 2},
            },
        },
    )

    with stubber:
        response = agent_client.retrieve(
            knowledgeBaseId="kb-12345678",
            retrievalQuery={"text": "What is OpenTelemetry?"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 2},
            },
        )
        assert len(response["retrievalResults"]) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.RETRIEVAL.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_PROVIDER_NAME)
        == GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "kb-12345678"
    )
    top_k = span.attributes.get("gen_ai.retrieval.top_k")
    assert isinstance(top_k, int)
    assert top_k == 2
    assert (
        span.attributes.get(ServerAttributes.SERVER_ADDRESS)
        == "bedrock-agent-runtime.us-east-1.amazonaws.com"
    )

    # Content capture
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "What is OpenTelemetry?"
    )
    docs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS)
    )
    assert len(docs) == 2
    assert docs[0]["id"] == "s3://my-bucket/doc1.txt"
    assert docs[0]["content"] == "Document 1 content"
    assert isinstance(docs[0]["score"], float)
    assert docs[0]["score"] == 0.95
    assert docs[0]["metadata"] == {"author": "Alice"}
    assert docs[1]["id"] == "https://example.com/doc2"
    assert docs[1]["content"] == "Document 2 content"
    assert docs[1]["score"] == 0.85


def test_extract_retrieve_response_document_ids(
    tracer_provider,
    span_exporter,
    monkeypatch,
) -> None:
    """Document id resolution across response shapes.

    Driven through the extractor rather than a stubbed client: ``documentId``
    and newer location types are absent from the oldest supported botocore
    service model, so ``Stubber`` would reject them.
    """
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "SPAN_ONLY"
    )
    handler = TelemetryHandler(tracer_provider=tracer_provider)
    invocation = handler.retrieval(
        provider=GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
        data_source_id="kb-12345678",
    )

    extract_retrieve_response(
        {
            "retrievalResults": [
                {
                    "content": {"text": "Document 1 content"},
                    "documentId": "doc-1",
                    "location": {
                        "type": "S3",
                        "s3Location": {"uri": "s3://my-bucket/doc1.txt"},
                    },
                },
                {
                    "content": {"text": "Document 2 content"},
                    "location": {
                        "type": "GOOGLE_DRIVE",
                        "googleDriveLocation": {
                            "url": "https://drive.google.com/doc2"
                        },
                    },
                },
                {
                    "content": {"text": "Document 3 content"},
                    "location": {
                        "type": "SQL",
                        "sqlLocation": {"query": "SELECT 1"},
                    },
                },
            ]
        },
        invocation,
        capture_content=True,
    )
    invocation.stop()

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    docs = json.loads(
        spans[0].attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS)
    )
    assert len(docs) == 3
    # documentId wins over the location locator.
    assert docs[0]["id"] == "doc-1"
    # No documentId: fall back to the location locator.
    assert docs[1]["id"] == "https://drive.google.com/doc2"
    # sqlLocation carries a query, not a locator, so no id can be derived.
    assert "id" not in docs[2]
    assert docs[2]["content"] == "Document 3 content"


def test_retrieve_sync_no_content(
    agent_client,
    instrument_no_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber.add_response(
        "retrieve",
        service_response={
            "retrievalResults": [
                {
                    "content": {"text": "Document 1 content"},
                    "score": 0.9,
                }
            ],
        },
        expected_params={
            "knowledgeBaseId": "kb-12345678",
            "retrievalQuery": {"text": "Confidential query"},
        },
    )

    with stubber:
        agent_client.retrieve(
            knowledgeBaseId="kb-12345678",
            retrievalQuery={"text": "Confidential query"},
        )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT not in span.attributes
    assert GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS not in span.attributes
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "kb-12345678"
    )


def test_retrieve_sync_error(
    agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(agent_client)
    stubber.add_client_error(
        "retrieve",
        service_error_code="ResourceNotFoundException",
        service_message="Knowledge base not found",
        expected_params={
            "knowledgeBaseId": "kb-missing123",
            "retrievalQuery": {"text": "Test"},
        },
    )

    with stubber:
        with pytest.raises(ClientError):
            agent_client.retrieve(
                knowledgeBaseId="kb-missing123",
                retrievalQuery={"text": "Test"},
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) in (
        "ResourceNotFoundException",
        "botocore.errorfactory.ResourceNotFoundException",
    )


# --- Async invoke_agent and retrieve tests ---


@pytest.mark.asyncio
async def test_async_invoke_agent_with_content(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "async-session-123",
            "completion": _MockAsyncEventStream(
                [
                    {"chunk": {"bytes": b"Async "}},
                    {"chunk": {"bytes": b"agent response"}},
                ]
            ),
        },
        expected_params={
            "agentId": "async-agent-1",
            "agentAliasId": "async-alias-1",
            "sessionId": "async-session-123",
            "inputText": "Hello async agent",
        },
    )

    with stubber:
        response = await async_agent_client.invoke_agent(
            agentId="async-agent-1",
            agentAliasId="async-alias-1",
            sessionId="async-session-123",
            inputText="Hello async agent",
        )
        collected = [chunk async for chunk in response["completion"]]

    assert len(collected) == 2

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.INVOKE_AGENT.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_ID) == "async-agent-1"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_VERSION)
        == "async-alias-1"
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_CONVERSATION_ID)
        == "async-session-123"
    )
    assert (
        span.attributes.get(ServerAttributes.SERVER_ADDRESS)
        == "bedrock-agent-runtime.us-east-1.amazonaws.com"
    )

    input_msgs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_INPUT_MESSAGES)
    )
    assert input_msgs[0]["parts"][0]["content"] == "Hello async agent"

    output_msgs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_OUTPUT_MESSAGES)
    )
    assert output_msgs[0]["parts"][0]["content"] == "Async agent response"


@pytest.mark.asyncio
async def test_async_invoke_agent_no_content(
    async_agent_client,
    instrument_no_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "async-session-123",
            "completion": _MockAsyncEventStream(
                [{"chunk": {"bytes": b"Async agent response"}}]
            ),
        },
        expected_params={
            "agentId": "async-agent-1",
            "agentAliasId": "async-alias-1",
            "sessionId": "async-session-123",
            "inputText": "Hello async agent",
        },
    )

    with stubber:
        response = await async_agent_client.invoke_agent(
            agentId="async-agent-1",
            agentAliasId="async-alias-1",
            sessionId="async-session-123",
            inputText="Hello async agent",
        )
        async for _ in response["completion"]:
            pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_AGENT_ID) == "async-agent-1"
    )
    assert GenAIAttributes.GEN_AI_INPUT_MESSAGES not in span.attributes
    assert GenAIAttributes.GEN_AI_OUTPUT_MESSAGES not in span.attributes


@pytest.mark.asyncio
async def test_async_invoke_agent_error(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber.add_client_error(
        "invoke_agent",
        service_error_code="ResourceNotFoundException",
        service_message="Agent not found",
        http_status_code=404,
    )

    with stubber:
        with pytest.raises(ClientError):
            await async_agent_client.invoke_agent(
                agentId="async-agent-1",
                agentAliasId="async-alias-1",
                sessionId="async-session-123",
                inputText="Hello async agent",
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) in (
        "ResourceNotFoundException",
        "botocore.errorfactory.ResourceNotFoundException",
    )


@pytest.mark.asyncio
async def test_async_invoke_agent_stream_error(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "async-session-123",
            "completion": _FailingAsyncEventStream(
                [{"chunk": {"bytes": b"Async part 1"}}, {"fail": True}]
            ),
        },
        expected_params={
            "agentId": "async-agent-1",
            "agentAliasId": "async-alias-1",
            "sessionId": "async-session-123",
            "inputText": "Hello async agent",
        },
    )

    with stubber:
        response = await async_agent_client.invoke_agent(
            agentId="async-agent-1",
            agentAliasId="async-alias-1",
            sessionId="async-session-123",
            inputText="Hello async agent",
        )
        with pytest.raises(ConnectionError):
            async for _ in response["completion"]:
                pass

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "ConnectionError"


@pytest.mark.asyncio
async def test_async_invoke_agent_caller_error_during_stream(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber._validate_response = lambda *args, **kwargs: None
    stubber.add_response(
        "invoke_agent",
        service_response={
            "contentType": "application/json",
            "sessionId": "async-session-123",
            "completion": _MockAsyncEventStream(
                [
                    {"chunk": {"bytes": b"part 1"}},
                    {"chunk": {"bytes": b"part 2"}},
                ]
            ),
        },
        expected_params={
            "agentId": "async-agent-1",
            "agentAliasId": "async-alias-1",
            "sessionId": "async-session-123",
            "inputText": "Hello async agent",
        },
    )

    with stubber:
        response = await async_agent_client.invoke_agent(
            agentId="async-agent-1",
            agentAliasId="async-alias-1",
            sessionId="async-session-123",
            inputText="Hello async agent",
        )
        with pytest.raises(RuntimeError):
            async with response["completion"] as stream:
                async for _ in stream:
                    raise RuntimeError("caller error")

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) == "RuntimeError"


@pytest.mark.asyncio
async def test_async_retrieve_with_content(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber.add_response(
        "retrieve",
        service_response={
            "retrievalResults": [
                {
                    "content": {"text": "Async doc content"},
                    "score": 0.98,
                    "location": {
                        "type": "S3",
                        "s3Location": {"uri": "s3://my-bucket/async.txt"},
                    },
                }
            ]
        },
        expected_params={
            "knowledgeBaseId": "async-kb-12345",
            "retrievalQuery": {"text": "Async retrieval query"},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {"numberOfResults": 1},
            },
        },
    )

    with stubber:
        response = await async_agent_client.retrieve(
            knowledgeBaseId="async-kb-12345",
            retrievalQuery={"text": "Async retrieval query"},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 1},
            },
        )

    assert len(response["retrievalResults"]) == 1

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == GenAIAttributes.GenAiOperationNameValues.RETRIEVAL.value
    )
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_DATA_SOURCE_ID)
        == "async-kb-12345"
    )
    assert span.attributes.get("gen_ai.retrieval.top_k") == 1
    assert (
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_QUERY_TEXT)
        == "Async retrieval query"
    )
    docs = json.loads(
        span.attributes.get(GenAIAttributes.GEN_AI_RETRIEVAL_DOCUMENTS)
    )
    assert len(docs) == 1
    assert docs[0]["content"] == "Async doc content"
    assert docs[0]["id"] == "s3://my-bucket/async.txt"
    assert docs[0]["score"] == 0.98


@pytest.mark.asyncio
async def test_async_retrieve_error(
    async_agent_client,
    instrument_with_content,
    span_exporter,
) -> None:
    stubber = Stubber(async_agent_client)
    stubber.add_client_error(
        "retrieve",
        service_error_code="ValidationException",
        service_message="Invalid KB ID",
        http_status_code=400,
    )

    with stubber:
        with pytest.raises(ClientError):
            await async_agent_client.retrieve(
                knowledgeBaseId="invalid-kb-12345",
                retrievalQuery={"text": "Query"},
            )

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes.get(ErrorAttributes.ERROR_TYPE) in (
        "ValidationException",
        "botocore.errorfactory.ValidationException",
    )
