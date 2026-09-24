# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: model chat inference for Agno."""

from __future__ import annotations

from typing import Any

from agno.models.message import Message
from agno.models.response import ModelResponse
from tests.mock_model import MockModel

try:
    from agno.models.message import MessageMetrics
except ImportError:
    from agno.models.message import Metrics as MessageMetrics

from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class _ConformanceChatModel(MockModel):
    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return ModelResponse(
            content="Conformance model response",
            provider_data={
                "model": "gpt-4o-2024-08-06",
                "id": "chatcmpl-conformance",
            },
            response_usage=MessageMetrics(
                input_tokens=12,
                output_tokens=6,
            ),
        )


class ChatScenario(Scenario):
    expected_spans = {"chat": 1}
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
            model = _ConformanceChatModel(id="gpt-4o", provider="OpenAI")
            model.response(
                messages=[
                    Message(role="user", content="hello chat conformance")
                ]
            )
