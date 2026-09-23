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
    def __init__(self, text: str, pid: str, score: float) -> None:
        self.long_text = text
        self.pid = pid
        self.score = score


class _DummyRM:
    def __call__(self, query: str, k: int = 3, **kwargs: Any) -> list[Any]:
        return [
            _DummyPassage(
                f"Passage {i} for {query}",
                pid=f"doc-{i}",
                score=round(0.95 - (i * 0.1), 2),
            )
            for i in range(k)
        ]


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


class ColBERTv2Scenario(Scenario):
    expected_spans = {"retrieval": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        from unittest.mock import patch

        from dspy.dsp.utils import dotdict

        def fake_get_request(url: str, query: str, k: int) -> list[Any]:
            return [
                dotdict(
                    {
                        "long_text": f"ColBERT result {i} for {query}",
                        "pid": 100 + i,
                        "score": round(0.95 - (i * 0.1), 2),
                    }
                )
                for i in range(k)
            ]

        colbert = dspy.ColBERTv2(
            url="http://colbert.example.com:8893/api/search"
        )
        with patch(
            "dspy.dsp.colbertv2.colbertv2_get_request",
            side_effect=fake_get_request,
        ):
            with instrument(
                DSPyInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                colbert("What is OpenTelemetry?", k=2)


class EmbeddingsScenario(Scenario):
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
        from dspy.retrievers.embeddings import EmbeddingsWithScores

        emb_retriever = EmbeddingsWithScores.__new__(EmbeddingsWithScores)
        emb_retriever.k = 1
        emb_retriever.corpus = [
            "OpenTelemetry is an observability framework.",
            "DSPy is a declarative framework for language models.",
        ]
        emb_retriever.search_fn = lambda query: (
            ["OpenTelemetry is an observability framework."],
            [0],
            [0.98],
        )
        with instrument(
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            emb_retriever("What is OpenTelemetry?")
