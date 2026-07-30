# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain retrieval via VectorStoreRetriever."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from opentelemetry.instrumentation.genai.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _FakeRetriever(BaseRetriever):
    """In-memory retriever that returns fixed documents without network calls."""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content="Paris is the capital of France.",
                id="doc-1",
                metadata={"source": "wiki"},
            ),
            Document(
                page_content="The Eiffel Tower is located in Paris.",
                id="doc-2",
                metadata={"source": "wiki"},
            ),
        ]

    def _get_ls_params(self, **kwargs: Any) -> Any:
        params = super()._get_ls_params(**kwargs)
        params["ls_vector_store_provider"] = "FakeVectorStore"
        return params


class RetrievalScenario(Scenario):
    expected_spans = {"retrieval": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        # _FakeRetriever is in-memory and has no backing server.
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
            LangChainInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            retriever = _FakeRetriever()
            # No VCR cassette needed — _FakeRetriever makes no network calls.
            retriever.invoke("What is the capital of France?")

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        operations = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.operation.name"
        ]
        assert "retrieval" in operations, (
            f"Expected a retrieval span; saw operations {operations}"
        )
