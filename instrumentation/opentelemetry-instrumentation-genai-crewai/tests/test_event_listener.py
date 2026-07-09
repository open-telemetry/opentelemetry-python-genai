# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CrewAI LLM event listener.

Drives the listener handlers directly with real CrewAI event objects —
deterministic, unlike the event bus's thread-pool dispatch, which the
conformance test covers end to end.
"""

from __future__ import annotations

from typing import Any

import pytest
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)

from opentelemetry import context as context_api
from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode


@pytest.fixture
def listener(instrument_crewai: CrewAIInstrumentor) -> Any:
    return instrument_crewai._event_listener


def _started_event(**overrides: Any) -> LLMCallStartedEvent:
    kwargs: dict[str, Any] = {
        "call_id": "call-1",
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
    }
    kwargs.update(overrides)
    return LLMCallStartedEvent(**kwargs)


def _completed_event(
    started: LLMCallStartedEvent | None, **overrides: Any
) -> LLMCallCompletedEvent:
    kwargs: dict[str, Any] = {
        "call_id": "call-1",
        "model": "gpt-4o-mini",
        "started_event_id": started.event_id if started else "unknown",
        "response": "Paris is the capital of France.",
        "call_type": LLMCallType.LLM_CALL,
        "usage": {"prompt_tokens": 89, "completion_tokens": 8},
        "finish_reason": "stop",
        "response_id": "chatcmpl-test123",
    }
    kwargs.update(overrides)
    return LLMCallCompletedEvent(**kwargs)


def test_completed_llm_call_emits_chat_span(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    started = _started_event()
    listener._on_llm_started(None, started)
    listener._on_llm_completed(None, _completed_event(started))

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    attributes = span.attributes
    assert attributes is not None
    assert (
        attributes[GenAI.GEN_AI_OPERATION_NAME]
        == GenAI.GenAiOperationNameValues.CHAT.value
    )
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "crewai"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    assert attributes[GenAI.GEN_AI_RESPONSE_ID] == "chatcmpl-test123"
    assert isinstance(attributes[GenAI.GEN_AI_RESPONSE_ID], str)
    assert tuple(attributes[GenAI.GEN_AI_RESPONSE_FINISH_REASONS]) == ("stop",)
    assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 89
    assert isinstance(attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS], int)
    assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
    assert isinstance(attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS], int)


def test_span_only_mode_captures_messages(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    started = _started_event()
    listener._on_llm_started(None, started)
    listener._on_llm_completed(None, _completed_event(started))

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes
    assert attributes is not None
    input_messages = attributes[GenAI.GEN_AI_INPUT_MESSAGES]
    assert isinstance(input_messages, str)
    assert "What is the capital of France?" in input_messages
    output_messages = attributes[GenAI.GEN_AI_OUTPUT_MESSAGES]
    assert isinstance(output_messages, str)
    assert "Paris is the capital of France." in output_messages


def test_no_content_mode_omits_messages(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
    span_exporter: InMemorySpanExporter,
) -> None:
    with instrument(
        CrewAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        semconv="gen_ai_latest_experimental",
        content_capture="NO_CONTENT",
    ) as instrumentor:
        listener = instrumentor._event_listener
        started = _started_event()
        listener._on_llm_started(None, started)
        listener._on_llm_completed(None, _completed_event(started))

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes
    assert attributes is not None
    assert GenAI.GEN_AI_INPUT_MESSAGES not in attributes
    assert GenAI.GEN_AI_OUTPUT_MESSAGES not in attributes


def test_completed_llm_call_records_metrics(
    listener: Any, metric_reader: InMemoryMetricReader
) -> None:
    started = _started_event()
    listener._on_llm_started(None, started)
    listener._on_llm_completed(None, _completed_event(started))

    metrics_data = metric_reader.get_metrics_data()
    assert metrics_data is not None
    metric_names = {
        metric.name
        for resource_metrics in metrics_data.resource_metrics
        for scope_metrics in resource_metrics.scope_metrics
        for metric in scope_metrics.metrics
    }
    assert gen_ai_metrics.GEN_AI_CLIENT_OPERATION_DURATION in metric_names
    assert gen_ai_metrics.GEN_AI_CLIENT_TOKEN_USAGE in metric_names


def test_usage_falls_back_to_response_payload(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    started = _started_event()
    listener._on_llm_started(None, started)
    listener._on_llm_completed(
        None,
        _completed_event(
            started,
            usage=None,
            response={
                "content": "Paris.",
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            },
        ),
    )

    (span,) = span_exporter.get_finished_spans()
    attributes = span.attributes
    assert attributes is not None
    assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 11
    assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 3


def test_failed_llm_call_records_error(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    started = _started_event()
    listener._on_llm_started(None, started)
    listener._on_llm_failed(
        None,
        LLMCallFailedEvent(
            call_id="call-1",
            model="gpt-4o-mini",
            started_event_id=started.event_id,
            error="boom",
        ),
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attributes = span.attributes
    assert attributes is not None
    # The event carries a plain string; the listener wraps it in RuntimeError.
    assert attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
    assert isinstance(attributes[error_attributes.ERROR_TYPE], str)


def test_completed_without_started_is_ignored(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    listener._on_llm_completed(None, _completed_event(None))

    assert not span_exporter.get_finished_spans()


def test_suppressed_instrumentation_emits_no_span(
    listener: Any, span_exporter: InMemorySpanExporter
) -> None:
    started = _started_event()
    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )
    try:
        listener._on_llm_started(None, started)
    finally:
        context_api.detach(token)
    listener._on_llm_completed(None, _completed_event(started))

    assert not span_exporter.get_finished_spans()


def test_shutdown_fails_open_invocations(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener
    listener._on_llm_started(None, _started_event())

    instrument_crewai.uninstrument()

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    attributes = span.attributes
    assert attributes is not None
    assert attributes[error_attributes.ERROR_TYPE] == "RuntimeError"
