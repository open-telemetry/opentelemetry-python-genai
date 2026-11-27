# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: openai legacy completion (text_completion).

Runs one non-streaming and one streaming call, so the streaming timing
metrics (time-to-first-chunk and time-per-output-chunk) and the stream flag
are validated in addition to the duration and token usage metrics.
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument


class TextCompletionScenario(Scenario):
    expected_spans = {"text_completion": 2}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        stream_values = [
            attr["value"]
            for entry in report["samples"]
            if "span" in entry
            for attr in entry["span"]["attributes"]
            if attr["name"] == "gen_ai.request.stream"
        ]
        assert stream_values == [True], (
            "only the streaming completion should set gen_ai.request.stream; "
            f"saw {stream_values}"
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
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette("test_completion_with_content.yaml"):
                OpenAI().completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=16,
                )
            with vcr.use_cassette("test_completion_streaming.yaml"):
                stream = OpenAI().completions.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt="Say this is a test",
                    max_tokens=16,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                for _ in stream:
                    pass
