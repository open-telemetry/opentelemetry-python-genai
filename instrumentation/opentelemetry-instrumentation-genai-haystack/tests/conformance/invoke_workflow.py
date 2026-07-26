# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: Pipeline.run wrapping a chat-generator component."""

from __future__ import annotations

from typing import Any

from haystack import Pipeline
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.generators.chat.openai import OpenAIChatGenerator
from haystack.dataclasses.chat_message import ChatMessage

from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.test_util_genai.conformance import Scenario
from opentelemetry.test_util_genai.instrumentor import instrument

from ._known_gaps import MISSING_RESPONSE_ID


class WorkflowScenario(Scenario):
    expected_spans = {"invoke_workflow": 1, "chat": 1}
    expected_metrics = (
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    )
    # Unlike the standalone-call scenarios, Pipeline.run() calls warm_up() on
    # its components before running them, so the chat generator's SDK client
    # (and therefore server.address) is already constructed by the time our
    # wrapper runs -- no server.address gap here.
    expected_violations = (MISSING_RESPONSE_ID,)

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
            with vcr.use_cassette("workflow_conformance.yaml"):
                pipeline = Pipeline()
                pipeline.add_component("prompt_builder", ChatPromptBuilder())
                pipeline.add_component(
                    "llm", OpenAIChatGenerator(model="gpt-4o")
                )
                pipeline.connect("prompt_builder.prompt", "llm.messages")
                pipeline.run(
                    data={
                        "prompt_builder": {
                            "template_variables": {"location": "Berlin"},
                            "template": [
                                ChatMessage.from_system(
                                    "Answer concisely in one sentence."
                                ),
                                ChatMessage.from_user(
                                    "What country is {{location}} in?"
                                ),
                            ],
                        }
                    }
                )
