# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any

import pytest
from google.genai import Client
from google.genai.models import AsyncModels, Models
from google.genai.types import (
    EmbedContentResponse,
    GenerateContentResponse,
)

from opentelemetry import context, trace
from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    Interaction,
    InteractionsResource,
)
from opentelemetry.instrumentation.google_genai.tool_call_wrapper import (
    wrapped_tool,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import Span
from opentelemetry.util.genai.handler import TelemetryHandler

REQUIRES_INTERACTIONS = pytest.mark.skipif(
    not _HAS_INTERACTIONS,
    reason="Interactions are not supported in this version of google-genai",
)


def _interaction() -> Any:
    return Interaction.model_construct(
        id="response-id", status="completed", steps=[]
    )


_CASES: dict[str, tuple[type, str, Callable[[], Any], Callable[..., Any]]] = {
    "generate_content": (
        Models,
        "generate_content",
        lambda: GenerateContentResponse(candidates=[]),
        lambda c: c.models.generate_content(
            model="gemini-2.5-flash", contents="Hello"
        ),
    ),
    "embed_content": (
        Models,
        "embed_content",
        lambda: EmbedContentResponse(embeddings=[]),
        lambda c: c.models.embed_content(
            model="text-embedding-004", contents="Hello"
        ),
    ),
    "interactions.create": (
        InteractionsResource,
        "create",
        _interaction,
        lambda c: c.interactions.create(
            model="gemini-2.5-flash", input="Hello"
        ),
    ),
    "interactions.get": (
        InteractionsResource,
        "get",
        _interaction,
        lambda c: c.interactions.get("response-id"),
    ),
}

_ASYNC_CASES: dict[
    str, tuple[type, str, Callable[[], Any], Callable[..., Any]]
] = {
    "generate_content": (
        AsyncModels,
        "generate_content",
        _CASES["generate_content"][2],
        lambda c: c.aio.models.generate_content(
            model="gemini-2.5-flash", contents="Hello"
        ),
    ),
    "embed_content": (
        AsyncModels,
        "embed_content",
        _CASES["embed_content"][2],
        lambda c: c.aio.models.embed_content(
            model="text-embedding-004", contents="Hello"
        ),
    ),
    "interactions.create": (
        AsyncInteractionsResource,
        "create",
        _interaction,
        lambda c: c.aio.interactions.create(
            model="gemini-2.5-flash", input="Hello"
        ),
    ),
    "interactions.get": (
        AsyncInteractionsResource,
        "get",
        _interaction,
        lambda c: c.aio.interactions.get("response-id"),
    ),
}

_OPERATIONS = [
    pytest.param("generate_content"),
    pytest.param("embed_content"),
    pytest.param("interactions.create", marks=REQUIRES_INTERACTIONS),
    pytest.param("interactions.get", marks=REQUIRES_INTERACTIONS),
]


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_span_is_current_only_during_the_call(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    operation: str,
) -> None:
    resource_type, method_name, make_response, call_sdk = _CASES[operation]
    seen: list[Span] = []
    original = getattr(resource_type, method_name)

    @wraps(original)
    def call(*args: object, **kwargs: object) -> Any:
        seen.append(trace.get_current_span())
        return make_response()

    monkeypatch.setattr(resource_type, method_name, call)
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="NO_CONTENT",
    ):
        before = context.get_current()
        call_sdk(Client(api_key="test-key"))
        assert context.get_current() is before

    (span,) = span_exporter.get_finished_spans()
    assert [s.get_span_context() for s in seen] == [span.context]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", _OPERATIONS)
async def test_async_span_is_current_only_during_the_call(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
    operation: str,
) -> None:
    resource_type, method_name, make_response, call_sdk = _ASYNC_CASES[
        operation
    ]
    seen: list[Span] = []
    original = getattr(resource_type, method_name)

    @wraps(original)
    async def call(*args: object, **kwargs: object) -> Any:
        seen.append(trace.get_current_span())
        return make_response()

    monkeypatch.setattr(resource_type, method_name, call)
    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
        content_capture="NO_CONTENT",
    ):
        before = context.get_current()
        await call_sdk(Client(api_key="test-key"))
        assert context.get_current() is before

    (span,) = span_exporter.get_finished_spans()
    assert [s.get_span_context() for s in seen] == [span.context]


def test_tool_span_is_current_only_during_the_tool(
    tracer_provider: TracerProvider,
    meter_provider: MeterProvider,
    logger_provider: LoggerProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    seen: list[Span] = []

    def get_weather(city: str) -> str:
        seen.append(trace.get_current_span())
        return "sunny"

    handler = TelemetryHandler(
        tracer_provider=tracer_provider,
        meter_provider=meter_provider,
        logger_provider=logger_provider,
    )
    tool = wrapped_tool(get_weather, handler)
    before = context.get_current()
    tool("Paris")
    assert context.get_current() is before

    (span,) = span_exporter.get_finished_spans()
    assert [s.get_span_context() for s in seen] == [span.context]
