# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: embedding generation for Agno."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agno.knowledge.embedder.openai import OpenAIEmbedder

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


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
        mock_entry = MagicMock(embedding=[0.1, 0.2, 0.3, 0.4])
        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = {
            "input_tokens": 8,
            "model": "text-embedding-3-small",
        }
        mock_resp = MagicMock(data=[mock_entry], usage=mock_usage)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_resp

        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            embedder = OpenAIEmbedder(
                api_key="fake",
                id="text-embedding-3-small",
                openai_client=mock_client,
            )
            embedder.get_embedding_and_usage(
                "conformance test embedding input"
            )
