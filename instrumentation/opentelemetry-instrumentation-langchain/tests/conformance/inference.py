# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain chat (inference) via ChatOpenAI."""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from opentelemetry.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = ("chat",)
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
        key_override = (
            {}
            if os.getenv("OPENAI_API_KEY")
            else {"OPENAI_API_KEY": "test_openai_api_key"}
        )
        with mock.patch.dict(os.environ, key_override):
            with instrument(
                LangChainInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                semconv="gen_ai_latest_experimental",
                content_capture="SPAN_ONLY",
            ):
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    max_tokens=100,
                    top_p=0.9,
                    frequency_penalty=0.5,
                    presence_penalty=0.5,
                    stop_sequences=["\n", "Human:", "AI:"],
                    seed=100,
                )
                with vcr.use_cassette("inference_conformance.yaml"):
                    llm.invoke(
                        [
                            SystemMessage(
                                content="You are a helpful assistant!"
                            ),
                            HumanMessage(
                                content="What is the capital of France?"
                            ),
                        ]
                    )
