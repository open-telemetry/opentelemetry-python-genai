# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: OpenAI Responses API server-side compaction."""

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

COMPACTION_MODEL = "gpt-5.4"
# Well past ``compact_threshold`` so the server compacts during the request.
COMPACTION_TRIGGER_INPUT = (
    "Remember this sentence for later. " * 400 + "Now say this is a test."
)


class ResponsesCompactionScenario(Scenario):
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
            OpenAIInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            with vcr.use_cassette(
                "test_responses_create_records_server_compaction[content_mode0].yaml"
            ):
                OpenAI().responses.create(
                    model=COMPACTION_MODEL,
                    instructions="You are a helpful assistant.",
                    input=COMPACTION_TRIGGER_INPUT,
                    extra_body={
                        "context_management": [
                            {"type": "compaction", "compact_threshold": 1000}
                        ]
                    },
                )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)

        (span,) = [
            entry["span"]
            for entry in report["samples"]
            if "span" in entry
            and _attr(entry["span"], "gen_ai.operation.name") == "chat"
        ]
        assert _attr(span, "gen_ai.conversation.compacted") is True


def _attr(span: dict[str, Any], name: str) -> Any:
    for attr in span["attributes"]:
        if attr["name"] == name:
            return attr["value"]
    return None
