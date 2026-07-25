# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAIChatGenerator.run (chat inference)."""

from __future__ import annotations

from typing import Any

from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = {"chat": 1}
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
        with instrument(
            HaystackInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("inference_conformance.yaml"):
                OpenAIChatGenerator(model="gpt-4o").run(
                    messages=[
                        ChatMessage.from_system(
                            "Answer user questions succinctly"
                        ),
                        ChatMessage.from_assistant(
                            "What can I help you with?"
                        ),
                        ChatMessage.from_user(
                            "Who won the World Cup in 2022? Answer in one word."
                        ),
                    ]
                )
