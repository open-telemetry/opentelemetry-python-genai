# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAIDocumentEmbedder.run (embeddings)."""

from __future__ import annotations

from typing import Any

from haystack import Document
from haystack.components.embedders.openai_document_embedder import (
    OpenAIDocumentEmbedder,
)

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class EmbeddingScenario(Scenario):
    expected_spans = {"embeddings": 1}
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
            HaystackInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("embedding_conformance.yaml"):
                OpenAIDocumentEmbedder(model="text-embedding-3-small").run(
                    documents=[
                        Document(
                            content="Argentina won the World Cup in 2022."
                        ),
                        Document(content="France won the World Cup in 2018."),
                    ]
                )
