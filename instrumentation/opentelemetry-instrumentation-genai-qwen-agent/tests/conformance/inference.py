# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: a chat operation via qwen-agent's BaseChatModel."""

from __future__ import annotations

from typing import Any

from qwen_agent.llm import get_chat_model

from opentelemetry.instrumentation.genai.qwen_agent import (
    QwenAgentInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


class InferenceScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # qwen-agent does not expose the resolved model service endpoint, so
    # server.address cannot be populated on chat spans.
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
            QwenAgentInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            llm = get_chat_model(
                {
                    "model": "qwen-max",
                    "model_type": "qwen_dashscope",
                    # use_raw_api only supports stream=True on newer
                    # qwen-agent versions.
                    "generate_cfg": {"use_raw_api": False},
                }
            )
            with vcr.use_cassette("inference.yaml"):
                list(
                    llm.chat(
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    "What is 2+2? Answer with just the number."
                                ),
                            }
                        ],
                        stream=False,
                    )
                )
