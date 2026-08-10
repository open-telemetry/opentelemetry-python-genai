# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API fetch by id."""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

RESPONSE_ID = "resp_0f4faba17dcd0f1e0069e2f3e4907881909179832ba1237025"


class ResponsesFetchScenario(Scenario):
    """Fetching a stored response emits a ``fetch_response`` span.

    The operation performs no inference, so no token usage is recorded and
    ``gen_ai.client.token.usage`` is intentionally absent.
    """

    expected_spans = {"fetch_response": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("responses_fetch_conformance.yaml"):
                OpenAI().responses.retrieve(RESPONSE_ID)
