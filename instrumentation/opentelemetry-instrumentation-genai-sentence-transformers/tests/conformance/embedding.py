# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: sentence-transformers embeddings."""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.genai.sentence_transformers import (
    SentenceTransformersInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from ..test_utils import sentence_transformer

# The model runs in this process: it listens on no socket, returns no resolved
# model name, and ``encode`` reports no token usage. semconv expects those
# three attributes on an embeddings span, so the scenario carries the gaps.
IN_PROCESS_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
        "server.address",
        "gen_ai.response.model",
        "gen_ai.usage.input_tokens",
    )
)


class EmbeddingScenario(Scenario):
    expected_spans = {"embeddings": 1}
    # No usage data, so no gen_ai.client.token.usage.
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = IN_PROCESS_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        # The model runs in-process, so there is no HTTP traffic and ``vcr``
        # is unused.
        with instrument(
            SentenceTransformersInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            sentence_transformer().encode(["hello world"])
