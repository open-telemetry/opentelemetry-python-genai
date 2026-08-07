# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)

from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.test_util_genai.instrumentor import instrument
from opentelemetry.trace import SpanKind, StatusCode


def _source() -> Any:
    return SimpleNamespace(
        provider="openai",
        model="gpt-4.1-nano",
        base_url="https://api.openai.com/v1",
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        seed=7,
        stop_sequences=["done"],
        frequency_penalty=0.1,
        presence_penalty=0.3,
        n=1,
    )


def _start() -> LLMCallStartedEvent:
    return LLMCallStartedEvent(
        call_id="call-1",
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate an expression",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=64,
        seed=7,
        stop_sequences=["done"],
        frequency_penalty=0.1,
        presence_penalty=0.3,
        n=1,
    )


def test_completed_call_emits_semantic_inference_telemetry(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener
    started = _start()
    listener._on_started(_source(), started)
    completed = LLMCallCompletedEvent(
        call_id="call-1",
        model="gpt-4.1-nano",
        started_event_id=started.event_id,
        response=SimpleNamespace(
            content="2 + 2 equals 4.",
            model="gpt-4.1-nano-2025-04-14",
            id="chatcmpl-test",
            finish_reason="stop",
        ),
        call_type=LLMCallType.LLM_CALL,
        usage={"prompt_tokens": 12, "completion_tokens": 8},
        finish_reason="stop",
        response_id="chatcmpl-test",
    )
    listener._on_completed(_source(), completed)

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "chat gpt-4.1-nano"
    assert span.kind == SpanKind.CLIENT
    assert span.status.status_code == StatusCode.UNSET
    attributes = span.attributes
    assert attributes is not None
    assert attributes[GenAI.GEN_AI_OPERATION_NAME] == "chat"
    assert attributes[GenAI.GEN_AI_PROVIDER_NAME] == "openai"
    assert attributes[GenAI.GEN_AI_REQUEST_MODEL] == "gpt-4.1-nano"
    assert attributes[GenAI.GEN_AI_RESPONSE_MODEL] == "gpt-4.1-nano-2025-04-14"
    assert attributes[GenAI.GEN_AI_RESPONSE_ID] == "chatcmpl-test"
    assert attributes[server_attributes.SERVER_ADDRESS] == "api.openai.com"
    assert attributes[GenAI.GEN_AI_REQUEST_TEMPERATURE] == 0.2
    assert attributes[GenAI.GEN_AI_REQUEST_TOP_P] == 0.9
    assert attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS] == 64
    assert isinstance(attributes[GenAI.GEN_AI_REQUEST_MAX_TOKENS], int)
    assert attributes[GenAI.GEN_AI_REQUEST_SEED] == 7
    assert tuple(attributes[GenAI.GEN_AI_REQUEST_STOP_SEQUENCES]) == ("done",)
    if getattr(completed, "usage", None) is not None:
        assert attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS] == 12
        assert isinstance(attributes[GenAI.GEN_AI_USAGE_INPUT_TOKENS], int)
        assert attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS] == 8
        assert isinstance(attributes[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS], int)
    assert isinstance(attributes[GenAI.GEN_AI_INPUT_MESSAGES], str)
    assert isinstance(attributes[GenAI.GEN_AI_OUTPUT_MESSAGES], str)
    assert isinstance(attributes[GenAI.GEN_AI_TOOL_DEFINITIONS], str)


def test_failed_call_ends_span_with_error(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    listener = instrument_crewai._event_listener
    started = _start()
    listener._on_started(_source(), started)
    listener._on_failed(
        _source(),
        LLMCallFailedEvent(
            call_id="call-1",
            model="gpt-4.1-nano",
            started_event_id=started.event_id,
            error="provider unavailable",
        ),
    )

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes is not None
    assert span.attributes[error_attributes.ERROR_TYPE].endswith(
        "RuntimeError"
    )


def test_unmatched_completion_is_ignored(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    instrument_crewai._event_listener._on_completed(
        _source(),
        LLMCallCompletedEvent(
            call_id="missing",
            model="gpt-4.1-nano",
            started_event_id="missing",
            response="ignored",
            call_type=LLMCallType.LLM_CALL,
        ),
    )
    assert not span_exporter.get_finished_spans()


def test_instrumentor_registers_with_crewai_event_bus(
    instrument_crewai: CrewAIInstrumentor,
    span_exporter: InMemorySpanExporter,
) -> None:
    started = _start()
    start_future = crewai_event_bus.emit(_source(), started)
    assert start_future is not None
    start_future.result(timeout=5)

    completed_future = crewai_event_bus.emit(
        _source(),
        LLMCallCompletedEvent(
            call_id="call-1",
            model="gpt-4.1-nano",
            started_event_id=started.event_id,
            response="2 + 2 equals 4.",
            call_type=LLMCallType.LLM_CALL,
            usage={"prompt_tokens": 12, "completion_tokens": 8},
            finish_reason="stop",
        ),
    )
    assert completed_future is not None
    completed_future.result(timeout=5)

    (span,) = span_exporter.get_finished_spans()
    assert span.attributes is not None
    assert span.attributes[GenAI.GEN_AI_PROVIDER_NAME] == "openai"


def test_explicit_completion_hook_receives_inference_content(
    tracer_provider: TracerProvider,
    logger_provider: LoggerProvider,
    meter_provider: MeterProvider,
) -> None:
    hook = MagicMock()
    with instrument(
        CrewAIInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        completion_hook=hook,
    ) as instrumentor:
        listener = instrumentor._event_listener
        started = _start()
        listener._on_started(_source(), started)
        listener._on_completed(
            _source(),
            LLMCallCompletedEvent(
                call_id="call-1",
                model="gpt-4.1-nano",
                started_event_id=started.event_id,
                response="2 + 2 equals 4.",
                call_type=LLMCallType.LLM_CALL,
                finish_reason="stop",
            ),
        )

    hook.on_completion.assert_called_once()
    call = hook.on_completion.call_args.kwargs
    assert call["inputs"]
    assert call["outputs"]
    assert call["tool_definitions"]
    assert call["span"] is not None
