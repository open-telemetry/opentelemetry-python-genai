# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios: anthropic chat via ``with_raw_response``.

``with_raw_response`` hands the caller the HTTP response and defers
deserialization to ``parse()``, so the instrumentation builds its telemetry on
a different path than a plain ``messages.create``. These scenarios check that
the path still produces a spec-conformant ``chat`` span and metric set, for
both the non-streaming and the streaming response.
"""

from __future__ import annotations

import os
from typing import Any
from unittest import mock

from anthropic import Anthropic

from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def _api_key_override() -> dict[str, str]:
    if os.getenv("ANTHROPIC_API_KEY"):
        return {}
    return {"ANTHROPIC_API_KEY": "test_anthropic_api_key"}


class InferenceRawResponseScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="missing_attribute",
            message_substring="gen_ai.usage.cache_creation.input_tokens",
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
        with mock.patch.dict(os.environ, _api_key_override()):
            with instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette(
                    "test_sync_messages_create_with_raw_response.yaml"
                ):
                    raw_response = (
                        Anthropic().messages.with_raw_response.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=100,
                            messages=[
                                {
                                    "role": "user",
                                    "content": "Say hello in one word.",
                                }
                            ],
                        )
                    )
                    raw_response.parse()


class InferenceRawResponseStreamingScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )
    expected_violations = (
        ExpectedViolation(
            advice_id="missing_attribute",
            message_substring="gen_ai.usage.cache_creation.input_tokens",
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
        with mock.patch.dict(os.environ, _api_key_override()):
            with instrument(
                AnthropicInstrumentor(),
                tracer_provider=tracer_provider,
                logger_provider=logger_provider,
                meter_provider=meter_provider,
                content_capture="SPAN_ONLY",
            ):
                with vcr.use_cassette(
                    "test_sync_messages_create_streaming_with_raw_response.yaml"
                ):
                    raw_response = (
                        Anthropic().messages.with_raw_response.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=100,
                            messages=[
                                {
                                    "role": "user",
                                    "content": "Say hello in one word.",
                                }
                            ],
                            stream=True,
                        )
                    )
                    for _ in raw_response.parse():
                        pass
