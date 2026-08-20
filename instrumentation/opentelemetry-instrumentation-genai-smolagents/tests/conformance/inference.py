# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenarios for the ``chat`` operation (plain, streamed, and with
tool definitions).

Every scenario drives an in-process runtime over the stubs in ``test_utils``,
since those are the only instrumented model classes.
"""

from __future__ import annotations

from typing import Any

from opentelemetry.instrumentation.genai.smolagents import (
    SmolagentsInstrumentor,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test.weaver_live_check import LiveCheckReport
from opentelemetry.test_util_genai.conformance import (
    ExpectedViolation,
    Scenario,
)
from opentelemetry.test_util_genai.instrumentor import instrument

from ..test_utils import (
    MESSAGES,
    GetWeatherTool,
    transformers_model,
)
from ._helpers import attr, chat_spans

# An in-process runtime returns only the generated text: no request id, no
# resolved model name, no stop reason. It also listens on no socket. semconv
# expects those four attributes on a chat span, so every scenario here carries
# the same gaps.
IN_PROCESS_GAPS = tuple(
    ExpectedViolation(
        advice_id="genai_expected_attribute_missing",
        message_substring=attribute,
    )
    for attribute in (
        "server.address",
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.response.finish_reasons",
    )
)


class ChatScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    expected_violations = IN_PROCESS_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            transformers_model().generate(messages=MESSAGES)


class StreamedChatScenario(Scenario):
    expected_spans = {"chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    )
    expected_violations = IN_PROCESS_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            model = transformers_model(stream_chunks=["In ", "Paris"])
            list(model.generate_stream(messages=MESSAGES))

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        for span in chat_spans(report):
            assert attr(span, "gen_ai.request.stream") is True, (
                "expected gen_ai.request.stream on a streamed chat span"
            )


class ToolDefinitionsScenario(Scenario):
    """A ``chat`` call that offers tools to the model.

    The in-process runtimes put the tool definitions in the prompt but never
    return tool calls (the agent parses those out of the generated text), so
    this covers the request side only.
    """

    expected_spans = {"chat": 1}
    expected_metrics = ("gen_ai.client.operation.duration",)
    expected_violations = IN_PROCESS_GAPS

    def run(
        self,
        *,
        tracer_provider: TracerProvider,
        meter_provider: MeterProvider,
        logger_provider: LoggerProvider,
        vcr: Any,
    ) -> None:
        with instrument(
            SmolagentsInstrumentor(),
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            content_capture="SPAN_ONLY",
        ):
            transformers_model().generate(
                messages=MESSAGES, tools_to_call_from=[GetWeatherTool()]
            )

    def validate(self, report: LiveCheckReport) -> None:
        super().validate(report)
        for span in chat_spans(report):
            assert attr(span, "gen_ai.tool.definitions"), (
                "expected gen_ai.tool.definitions on the chat span"
            )
