# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: language model (inference) execution for DSPy."""

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


class AttrDict(dict):
    """Dictionary supporting attribute access for cross-DSPy-version compatibility."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class FakeLM(dspy.LM):
    """Test helper inheriting directly from dspy.LM."""

    def __init__(
        self,
        responses: list[str] | None = None,
        model: str = "openai/gpt-4o",
        model_type: str = "chat",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key="fake-api-key",
            model_type=model_type,
            **kwargs,
        )
        self._responses = list(responses or ["Paris"])
        self._idx = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        resp_text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        choice = AttrDict(
            {
                "message": AttrDict(
                    {
                        "content": str(resp_text),
                        "role": "assistant",
                        "reasoning_content": None,
                        "tool_calls": None,
                    }
                ),
                "finish_reason": "stop",
            }
        )
        return AttrDict(
            {
                "choices": [choice],
                "model": "gpt-4o-2024-05-13",
                "id": "chatcmpl-123",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        )

    async def aforward(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)


class LMScenario(Scenario):
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
            DSPyInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            lm = FakeLM(responses=["Paris"])
            lm("What is the capital of France?")
