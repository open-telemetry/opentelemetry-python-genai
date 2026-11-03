# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Embedder execution for DSPy."""

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


def _dummy_embedder(texts: list[str], **kwargs: Any) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class EmbedderScenario(Scenario):
    expected_spans = {"embeddings": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.model",
        ),
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.usage.input_tokens",
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
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            embedder = dspy.Embedder(
                _dummy_embedder,
                caching=False,
                encoding_format="float",
                api_base="https://api.openai.com:443/v1",
            )
            embedder(["hello", "world"])


class HostedEmbedderScenario(Scenario):
    expected_spans = {"embeddings": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        def _fake_embedding(
            model: str, input: list[str], **kwargs: Any
        ) -> SimpleNamespace:
            return SimpleNamespace(
                model="text-embedding-3-small-2024",
                data=[{"embedding": [0.1, 0.2, 0.3, 0.4, 0.5]} for _ in input],
                usage=SimpleNamespace(prompt_tokens=len(input) * 3),
            )

        fake_litellm = SimpleNamespace(
            cache=None,
            embedding=MagicMock(side_effect=_fake_embedding),
        )

        with (
            patch.object(
                dspy.clients.embedding,
                "_get_litellm",
                return_value=fake_litellm,
                create=True,
            ),
            patch.object(
                dspy.clients.embedding,
                "litellm",
                fake_litellm,
                create=True,
            ),
            instrument(
                DSPyInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ),
        ):
            embedder = dspy.Embedder(
                "openai/text-embedding-3-small",
                caching=False,
                encoding_format="float",
                api_base="https://api.openai.com:443/v1",
            )
            embedder(["hello", "world"])
