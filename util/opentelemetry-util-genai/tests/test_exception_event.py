# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``gen_ai.client.operation.exception`` event."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    exception_attributes,
)
from opentelemetry.trace import get_current_span
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.stream import AbandonedStreamError
from opentelemetry.util.genai.types import Error

EVENT_NAME = "gen_ai.client.operation.exception"

CLIENT_OPERATIONS: dict[str, Callable[[TelemetryHandler], Any]] = {
    "inference": lambda h: h.inference("provider", request_model="model"),
    "embedding": lambda h: h.embedding("provider", request_model="model"),
    "retrieval": lambda h: h.retrieval(provider="provider"),
    "fetch_response": lambda h: h.fetch_response(
        "provider", response_id="resp_1"
    ),
    "invoke_remote_agent": lambda h: h.invoke_remote_agent("provider"),
}

INTERNAL_OPERATIONS: dict[str, Callable[[TelemetryHandler], Any]] = {
    "tool": lambda h: h.tool("get_weather"),
    "workflow": lambda h: h.workflow("workflow"),
    "invoke_local_agent": lambda h: h.invoke_local_agent(agent_name="agent"),
}


class RateLimitError(Exception):
    pass


def _raise_rate_limit() -> None:
    raise RateLimitError("rate limit reached")


class _Telemetry:
    def __init__(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        )

    def exception_events(self) -> list[Any]:
        return [
            log.log_record
            for log in self.log_exporter.get_finished_logs()
            if log.log_record.event_name == EVENT_NAME
        ]

    def only_span(self) -> Any:
        (span,) = self.span_exporter.get_finished_spans()
        return span


@pytest.fixture
def telemetry() -> _Telemetry:
    return _Telemetry()


@pytest.mark.parametrize(
    "operation", CLIENT_OPERATIONS.values(), ids=CLIENT_OPERATIONS.keys()
)
def test_failed_client_operation_emits_an_exception_event(
    telemetry: _Telemetry, operation: Callable[[TelemetryHandler], Any]
) -> None:
    with pytest.raises(RateLimitError):
        with operation(telemetry.handler):
            _raise_rate_limit()

    (event,) = telemetry.exception_events()
    span = telemetry.only_span()
    assert event.severity_number == SeverityNumber.WARN
    assert event.trace_id == span.context.trace_id
    assert event.span_id == span.context.span_id
    assert set(event.attributes) == {
        exception_attributes.EXCEPTION_TYPE,
        exception_attributes.EXCEPTION_MESSAGE,
        exception_attributes.EXCEPTION_STACKTRACE,
    }
    assert event.attributes[exception_attributes.EXCEPTION_TYPE] == (
        f"{RateLimitError.__module__}.{RateLimitError.__qualname__}"
    )
    assert (
        event.attributes[exception_attributes.EXCEPTION_MESSAGE]
        == "rate limit reached"
    )
    stacktrace = event.attributes[exception_attributes.EXCEPTION_STACKTRACE]
    assert stacktrace.startswith("Traceback (most recent call last):")
    assert "_raise_rate_limit" in stacktrace
    assert stacktrace.rstrip().endswith("RateLimitError: rate limit reached")


@pytest.mark.parametrize(
    "operation", INTERNAL_OPERATIONS.values(), ids=INTERNAL_OPERATIONS.keys()
)
def test_failed_internal_operation_emits_no_exception_event(
    telemetry: _Telemetry, operation: Callable[[TelemetryHandler], Any]
) -> None:
    with pytest.raises(RateLimitError):
        with operation(telemetry.handler):
            _raise_rate_limit()

    assert telemetry.only_span().status.status_code == StatusCode.ERROR
    assert telemetry.exception_events() == []


def test_successful_operation_emits_no_exception_event(
    telemetry: _Telemetry,
) -> None:
    with telemetry.handler.inference("provider", request_model="model"):
        pass

    assert telemetry.exception_events() == []


def test_error_without_an_exception_emits_no_exception_event(
    telemetry: _Telemetry,
) -> None:
    # A failure reported by the provider without a raised exception has no
    # exception type, message, or stacktrace to record.
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(Error(type="response.failed", message="server error"))

    assert telemetry.only_span().status.status_code == StatusCode.ERROR
    assert telemetry.exception_events() == []


@pytest.mark.parametrize(
    "exception",
    [
        asyncio.CancelledError(),
        KeyboardInterrupt(),
        SystemExit(1),
        GeneratorExit(),
    ],
    ids=lambda e: type(e).__name__,
)
def test_base_exception_emits_no_exception_event(
    telemetry: _Telemetry, exception: BaseException
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(exception)

    assert telemetry.only_span().status.status_code == StatusCode.ERROR
    assert telemetry.exception_events() == []


def test_abandoned_stream_emits_no_exception_event(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(AbandonedStreamError())

    assert telemetry.only_span().status.status_code == StatusCode.ERROR
    assert telemetry.exception_events() == []


def test_exception_type_is_the_class_even_when_error_type_is_resolved(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference(
        "provider",
        request_model="m",
        error_type_resolver=lambda _: "rate_limited",
    )
    invocation.fail(RateLimitError("rate limit reached"))

    assert (
        telemetry.only_span().attributes[error_attributes.ERROR_TYPE]
        == "rate_limited"
    )
    (event,) = telemetry.exception_events()
    assert event.attributes[exception_attributes.EXCEPTION_TYPE] == (
        f"{RateLimitError.__module__}.{RateLimitError.__qualname__}"
    )


def test_builtin_exception_type_is_not_qualified(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(ConnectionError("connection refused"))

    (event,) = telemetry.exception_events()
    assert (
        event.attributes[exception_attributes.EXCEPTION_TYPE]
        == "ConnectionError"
    )


def test_exception_that_was_never_raised_has_a_stacktrace(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(ValueError("bad request"))

    (event,) = telemetry.exception_events()
    assert (
        event.attributes[exception_attributes.EXCEPTION_STACKTRACE]
        == "ValueError: bad request\n"
    )


def test_suppressed_inference_emits_no_exception_event(
    telemetry: _Telemetry,
) -> None:
    # The nested inference does not own a span, so the inference it runs
    # inside is the one that reports the failure.
    with telemetry.handler.inference("framework", request_model="m"):
        nested = telemetry.handler.inference("provider", request_model="m")
        nested.fail(RateLimitError("rate limit reached"))

    assert telemetry.exception_events() == []


def test_failing_twice_emits_one_exception_event(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    invocation.fail(RateLimitError("first"))
    invocation.fail(RateLimitError("second"))

    (event,) = telemetry.exception_events()
    assert event.attributes[exception_attributes.EXCEPTION_MESSAGE] == "first"


def test_disabled_logger_emits_no_exception_event(
    telemetry: _Telemetry,
) -> None:
    invocation = telemetry.handler.inference("provider", request_model="m")
    with (
        mock.patch.object(
            invocation._logger, "enabled", return_value=False
        ) as enabled,
        mock.patch.object(invocation._logger, "emit") as emit,
    ):
        invocation.fail(RateLimitError("rate limit reached"))

    enabled.assert_called_once()
    kwargs = enabled.call_args.kwargs
    assert kwargs["event_name"] == EVENT_NAME
    assert kwargs["severity_number"] == SeverityNumber.WARN
    assert get_current_span(kwargs["context"]) is invocation.span
    # The inference details event may still be emitted; the exception one is not.
    assert all(
        call.args[0].event_name != EVENT_NAME for call in emit.call_args_list
    )
    assert telemetry.only_span().status.status_code == StatusCode.ERROR
