# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval execution for DSPy."""

from __future__ import annotations

from typing import Any

import dspy

from opentelemetry.instrumentation.genai.dspy import DSPyInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _DummyPassage:
    def __init__(self, text: str) -> None:
        self.long_text = text


class _DummyRM:
    def __call__(self, query: str, k: int = 3, **kwargs: Any) -> list[Any]:
        return [_DummyPassage(f"Passage {i} for {query}") for i in range(k)]


class RetrieveScenario(Scenario):
    expected_spans = {"retrieval": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        dspy.settings.configure(rm=_DummyRM())
        with instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            retrieve = dspy.Retrieve(k=2)
            retrieve("What is OpenTelemetry?")
