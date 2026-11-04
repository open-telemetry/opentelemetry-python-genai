# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import nullcontext
from functools import wraps

import pytest
from google.genai import Client
from google.genai.models import AsyncModels, Models
from google.genai.types import GenerateContentResponse

from opentelemetry import context, trace
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    InteractionsResource,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import Span, StatusCode

REQUIRES_INTERACTIONS = pytest.mark.skipif(
    not _HAS_INTERACTIONS,
    reason="Interactions are not supported in this version of google-genai",
)


pytestmark = [
    pytest.mark.parametrize(
        "expected_operation",
        [
            pytest.param(
                "generate_content", id="models.generate_content_stream"
            ),
            pytest.param(
                "interactions.create",
                id="interactions.create-model",
                marks=REQUIRES_INTERACTIONS,
            ),
            pytest.param(
                "invoke_agent",
                id="interactions.create-agent",
                marks=REQUIRES_INTERACTIONS,
            ),
            pytest.param(
                "fetch_response",
                id="interactions.get",
                marks=REQUIRES_INTERACTIONS,
            ),
        ],
    ),
    pytest.mark.parametrize(
        "ending", ["drain", "close", "caller_error", "stream_error"]
    ),
]


def _assert_finished_spans(
    span_exporter: InMemorySpanExporter,
    sdk_span: Span,
    caller: Span,
    expected_operation: str,
    ending: str,
) -> None:
    invocation_span, caller_span = span_exporter.get_finished_spans()
    assert invocation_span.context == sdk_span.get_span_context()
    assert invocation_span.parent == caller.get_span_context()
    assert caller_span.context == caller.get_span_context()
    assert invocation_span.attributes is not None
    assert (
        invocation_span.attributes[gen_ai_attributes.GEN_AI_OPERATION_NAME]
        == expected_operation
    )
    failed = ending in ("caller_error", "stream_error")
    assert invocation_span.status.status_code == (
        StatusCode.ERROR if failed else StatusCode.UNSET
    )
    if failed:
        assert invocation_span.attributes[ERROR_TYPE] == "ConnectionError"
    else:
        assert ERROR_TYPE not in invocation_span.attributes


def test_sync_stream_context(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    expected_operation: str,
    ending: str,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    sdk_span: Span | None = None
    cleanup_spans: list[Span] = []
    error = ConnectionError("stream failed")
    chunk = (
        GenerateContentResponse(response_id="response-id", candidates=[])
        if expected_operation == "generate_content"
        else {"event_type": "content"}
    )

    def produce() -> Iterator[object]:
        try:
            for _ in range(2):
                assert trace.get_current_span() is sdk_span
                yield chunk
                if ending == "stream_error":
                    raise error
        finally:
            cleanup_spans.append(trace.get_current_span())

    resource_type = (
        Models
        if expected_operation == "generate_content"
        else InteractionsResource
    )
    method_name = (
        "generate_content_stream"
        if expected_operation == "generate_content"
        else "get"
        if expected_operation == "fetch_response"
        else "create"
    )
    original = getattr(resource_type, method_name)
    source = produce()

    @wraps(original)
    def call(*args: object, **kwargs: object) -> Iterator[object]:
        nonlocal sdk_span
        sdk_span = trace.get_current_span()
        return source

    monkeypatch.setattr(resource_type, method_name, call)
    with (
        instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="NO_CONTENT",
        ),
        tracer.start_as_current_span("caller") as caller,
    ):
        client = Client(api_key="test-key")
        before = context.get_current()
        if expected_operation == "generate_content":
            stream = client.models.generate_content_stream(
                model="gemini-2.5-flash", contents="Hello"
            )
        elif expected_operation == "fetch_response":
            stream = client.interactions.get("response-id", stream=True)
        elif expected_operation == "invoke_agent":
            stream = client.interactions.create(
                agent="test-agent", input="Hello", stream=True
            )
        else:
            stream = client.interactions.create(
                model="gemini-2.5-flash", input="Hello", stream=True
            )
        assert sdk_span is not None
        assert sdk_span.is_recording()
        assert context.get_current() is before
        received: list[object] = []

        with (
            pytest.raises(ConnectionError)
            if ending in ("caller_error", "stream_error")
            else nullcontext()
        ) as raised:
            with stream:
                for item in stream:
                    assert item is chunk
                    assert context.get_current() is before
                    assert sdk_span.is_recording()
                    received.append(item)
                    if ending == "caller_error":
                        raise error
                    if ending == "close":
                        break

        if raised is not None:
            assert raised.value is error
        assert len(received) == (2 if ending == "drain" else 1)
        assert cleanup_spans == [sdk_span]
        assert context.get_current() is before
        assert not sdk_span.is_recording()

    _assert_finished_spans(
        span_exporter, sdk_span, caller, expected_operation, ending
    )


@pytest.mark.asyncio
async def test_async_stream_context(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    expected_operation: str,
    ending: str,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)
    sdk_span: Span | None = None
    cleanup_spans: list[Span] = []
    error = ConnectionError("stream failed")
    chunk = (
        GenerateContentResponse(response_id="response-id", candidates=[])
        if expected_operation == "generate_content"
        else {"event_type": "content"}
    )

    async def produce() -> AsyncIterator[object]:
        try:
            for _ in range(2):
                assert trace.get_current_span() is sdk_span
                yield chunk
                if ending == "stream_error":
                    raise error
        finally:
            cleanup_spans.append(trace.get_current_span())

    resource_type = (
        AsyncModels
        if expected_operation == "generate_content"
        else AsyncInteractionsResource
    )
    method_name = (
        "generate_content_stream"
        if expected_operation == "generate_content"
        else "get"
        if expected_operation == "fetch_response"
        else "create"
    )
    original = getattr(resource_type, method_name)
    source = produce()

    @wraps(original)
    async def call(*args: object, **kwargs: object) -> AsyncIterator[object]:
        nonlocal sdk_span
        sdk_span = trace.get_current_span()
        return source

    monkeypatch.setattr(resource_type, method_name, call)
    with (
        instrument(
            GoogleGenAiSdkInstrumentor(),
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            content_capture="NO_CONTENT",
        ),
        tracer.start_as_current_span("caller") as caller,
    ):
        client = Client(api_key="test-key")
        before = context.get_current()
        if expected_operation == "generate_content":
            stream = await client.aio.models.generate_content_stream(
                model="gemini-2.5-flash", contents="Hello"
            )
        elif expected_operation == "fetch_response":
            stream = await client.aio.interactions.get(
                "response-id", stream=True
            )
        elif expected_operation == "invoke_agent":
            stream = await client.aio.interactions.create(
                agent="test-agent", input="Hello", stream=True
            )
        else:
            stream = await client.aio.interactions.create(
                model="gemini-2.5-flash", input="Hello", stream=True
            )
        assert sdk_span is not None
        assert sdk_span.is_recording()
        assert context.get_current() is before
        received: list[object] = []

        async def consume() -> None:
            try:
                async with stream:
                    async for item in stream:
                        assert item is chunk
                        assert context.get_current() is before
                        assert sdk_span.is_recording()
                        received.append(item)
                        if ending == "caller_error":
                            raise error
                        if ending == "close":
                            break
            finally:
                assert context.get_current() is before

        with (
            pytest.raises(ConnectionError)
            if ending in ("caller_error", "stream_error")
            else nullcontext()
        ) as raised:
            await asyncio.create_task(consume())

        if raised is not None:
            assert raised.value is error
        assert len(received) == (2 if ending == "drain" else 1)
        assert cleanup_spans == [sdk_span]
        assert context.get_current() is before
        assert not sdk_span.is_recording()

    _assert_finished_spans(
        span_exporter, sdk_span, caller, expected_operation, ending
    )
