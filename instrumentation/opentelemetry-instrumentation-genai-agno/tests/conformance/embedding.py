# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: embedding generation for Agno."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agno.knowledge.embedder.base import Embedder

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


@dataclass
class ConformanceEmbedder(Embedder):
    id: str = "text-embedding-3-small"
    provider: str = "openai"

    def get_embedding_and_usage(
        self, text: str
    ) -> tuple[list[float], dict[str, Any]]:
        return [0.1, 0.2, 0.3, 0.4], {
            "input_tokens": 8,
            "model": "text-embedding-3-small",
        }


class EmbeddingScenario(Scenario):
    expected_spans = {"embeddings": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
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
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            embedder = ConformanceEmbedder()
            embedder.get_embedding_and_usage(
                "conformance test embedding input"
            )
