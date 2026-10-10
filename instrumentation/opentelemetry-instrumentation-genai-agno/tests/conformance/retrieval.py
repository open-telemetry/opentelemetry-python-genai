# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval via Knowledge.search for Agno."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from agno.knowledge.knowledge import Document, Knowledge

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class RetrievalScenario(Scenario):
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
        with instrument(
            AgnoInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            kb = Knowledge(name="conformance-kb", max_results=2)
            kb.vector_db = MagicMock()
            kb.vector_db.name = "conformance_vector_db"
            kb.vector_db.provider = "pgvector"
            kb.vector_db.search.return_value = [
                Document(
                    content="OpenTelemetry provides observability standards.",
                    id="doc-1",
                    meta_data={"source": "docs"},
                    reranking_score=0.95,
                ),
                Document(
                    content="Agno provides multi-agent framework.",
                    id="doc-2",
                    meta_data={"source": "docs"},
                    reranking_score=0.88,
                ),
            ]
            kb.search(query="what is OpenTelemetry?", max_results=2)
