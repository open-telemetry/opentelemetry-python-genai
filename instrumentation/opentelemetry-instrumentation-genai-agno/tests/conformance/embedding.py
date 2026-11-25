# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: embedding generation for Agno."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from agno.knowledge.embedder.openai import OpenAIEmbedder
from openai.types import CreateEmbeddingResponse, Embedding

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_EMBEDDINGS_DIMENSION_COUNT,
    GEN_AI_USAGE_INPUT_TOKENS,
)
from opentelemetry.test.weaver_live_check import LiveCheckReport
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
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.model",
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
        responses = [
            CreateEmbeddingResponse(
                data=[
                    Embedding(
                        embedding=[0.1, 0.2],
                        index=index,
                        object="embedding",
                    )
                    for index in range(2)
                ],
                model="text-embedding-3-small",
                object="list",
                usage={"prompt_tokens": 7, "total_tokens": 7},
            ),
            CreateEmbeddingResponse(
                data=[
                    Embedding(
                        embedding=[0.3, 0.4],
                        index=0,
                        object="embedding",
                    )
                ],
                model="text-embedding-3-small",
                object="list",
                usage={"prompt_tokens": 4, "total_tokens": 4},
            ),
        ]
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(side_effect=responses)

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
                async_client=mock_client,
                batch_size=2,
            )
            asyncio.run(
                embedder.async_get_embeddings_batch_and_usage(
                    ["first", "second", "third"]
                )
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        attributes = {
            attr["name"]: attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
        }
        assert attributes[GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 2
        assert attributes[GEN_AI_USAGE_INPUT_TOKENS] == 11
