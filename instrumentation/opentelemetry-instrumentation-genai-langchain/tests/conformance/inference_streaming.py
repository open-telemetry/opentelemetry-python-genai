# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: langchain streaming chat (inference) via ChatOpenAI.

Exercises ``.stream()`` so the streaming timing metrics (time-to-first-chunk
and time-per-output-chunk) are emitted and validated alongside the duration and
token usage metrics.
"""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

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

from ._shared import span_attribute_values


class InferenceStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )
    # langchain can't populate server.address on chat spans.
    # gen_ai.response.id is absent because the langchain-openai streaming path
    # does not surface the id the provider sends on every chunk.
    expected_violations = (
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="server.address",
        ),
        ExpectedViolation(
            advice_id="genai_expected_attribute_missing",
            message_substring="gen_ai.response.id",
        ),
    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        stream_values = span_attribute_values(report, "gen_ai.request.stream")
        assert stream_values == [True], (
            "streaming chat should set gen_ai.request.stream=true on the chat "
            f"span; saw {stream_values}"
        )
        system_instructions = span_attribute_values(
            report, "gen_ai.system_instructions"
        )
        assert len(system_instructions) == 1, (
            "streaming chat span with a SystemMessage input should set "
            f"gen_ai.system_instructions once; saw {system_instructions}"
        )
        assert "You are a helpful assistant!" in system_instructions[0], (
            "gen_ai.system_instructions should carry the SystemMessage "
            f"content; got {system_instructions[0]}"
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
                    stream_usage=True,
                )
                with vcr.use_cassette("inference_streaming_conformance.yaml"):
                    for _ in llm.stream(
                        [
                            SystemMessage(
                                content="You are a helpful assistant!"
                            ),
                            HumanMessage(
                                content="What is the capital of France?"
                            ),
                        ]
                    ):
                        pass
