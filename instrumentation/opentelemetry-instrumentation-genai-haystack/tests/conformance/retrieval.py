# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: InMemoryBM25Retriever.run (retrieval).

Local/deterministic component -- no HTTP interaction, no cassette needed.
"""

from __future__ import annotations

from typing import Any

from haystack import Document
from haystack.components.retrievers.in_memory.bm25_retriever import (
    InMemoryBM25Retriever,
)
from haystack.document_stores.in_memory.document_store import (
    InMemoryDocumentStore,
)

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ._known_gaps import MISSING_SERVER_ADDRESS


class RetrievalScenario(Scenario):
    expected_spans = {"retrieval": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    # Unlike the generator/embedder scenarios, this isn't a lazy-client
    # timing issue -- InMemoryDocumentStore has no server at all.
    expected_violations = (MISSING_SERVER_ADDRESS,)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,  # noqa: ARG002 - unused; component makes no HTTP calls
    ) -> None:
        with instrument(
            HaystackInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            store = InMemoryDocumentStore()
            store.write_documents(
                [
                    Document(
                        content="Use pip to install Haystack's latest release"
                    )
                ]
            )
            InMemoryBM25Retriever(document_store=store).run(
                query="install Haystack", top_k=1
            )
