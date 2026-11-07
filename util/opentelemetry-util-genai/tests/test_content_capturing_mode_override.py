# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import unittest
from collections.abc import Callable
from unittest.mock import patch

from opentelemetry.context import get_current
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    InputMessage,
    TextPart,
)
from opentelemetry.util.genai.utils import override_content_capturing_mode

_INFERENCE_EVENT = "gen_ai.client.inference.operation.details"


def _user_message(text: str) -> list[InputMessage]:
    return [InputMessage(role="user", parts=[TextPart(content=text)])]


class TestOverrideContentCapturingMode(unittest.TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        self.logger_provider = LoggerProvider()
        self.logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )

    def _handler(self, configured_mode: str) -> TelemetryHandler:
        with patch.dict(
            os.environ,
            {
                OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: configured_mode
            },
        ):
            return TelemetryHandler(
                tracer_provider=self.tracer_provider,
                logger_provider=self.logger_provider,
            )

    @staticmethod
    def _chat(handler: TelemetryHandler, text: str) -> None:
        with handler.inference(
            "test-provider", request_model="test-model"
        ) as invocation:
            invocation.input_messages = _user_message(text)

    def _spans_with(self, attribute: str) -> list[bool]:
        return [
            attribute in (span.attributes or {})
            for span in self.span_exporter.get_finished_spans()
        ]

    def test_should_capture_content_follows_override(self) -> None:
        handler = self._handler("NO_CONTENT")
        with override_content_capturing_mode(ContentCapturingMode.SPAN_ONLY):
            self.assertTrue(handler.should_capture_content())
        self.assertFalse(handler.should_capture_content())

        handler = self._handler("SPAN_ONLY")
        with override_content_capturing_mode(ContentCapturingMode.NO_CONTENT):
            self.assertFalse(handler.should_capture_content())
        self.assertTrue(handler.should_capture_content())

    def test_capture_turned_on_for_one_request(self) -> None:
        handler = self._handler("NO_CONTENT")
        self._chat(handler, "before")
        with override_content_capturing_mode(ContentCapturingMode.SPAN_ONLY):
            self._chat(handler, "inside")
        self._chat(handler, "after")

        self.assertEqual(
            self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [False, True, False]
        )

    def test_capture_turned_off_for_one_request(self) -> None:
        handler = self._handler("SPAN_ONLY")
        self._chat(handler, "before")
        with override_content_capturing_mode(ContentCapturingMode.NO_CONTENT):
            self._chat(handler, "sensitive")
        self._chat(handler, "after")

        self.assertEqual(
            self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [True, False, True]
        )

    def test_override_mode_decides_where_content_is_recorded(self) -> None:
        cases = {
            ContentCapturingMode.NO_CONTENT: (False, False),
            ContentCapturingMode.SPAN_ONLY: (True, False),
            ContentCapturingMode.EVENT_ONLY: (False, True),
            ContentCapturingMode.SPAN_AND_EVENT: (True, True),
        }
        for mode, (on_span, on_event) in cases.items():
            with self.subTest(mode=mode.name):
                self.span_exporter.clear()
                self.log_exporter.clear()
                configured = (
                    "SPAN_AND_EVENT"
                    if mode is ContentCapturingMode.NO_CONTENT
                    else "NO_CONTENT"
                )
                handler = self._handler(configured)
                with override_content_capturing_mode(mode):
                    self._chat(handler, "hi")

                self.assertEqual(
                    self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [on_span]
                )
                events = [
                    log.log_record
                    for log in self.log_exporter.get_finished_logs()
                    if log.log_record.event_name == _INFERENCE_EVENT
                ]
                self.assertEqual(len(events), 1)
                self.assertEqual(
                    GenAI.GEN_AI_INPUT_MESSAGES
                    in (events[0].attributes or {}),
                    on_event,
                )

    def test_operation_keeps_the_mode_it_started_with(self) -> None:
        handler = self._handler("NO_CONTENT")
        with override_content_capturing_mode(ContentCapturingMode.SPAN_ONLY):
            invocation = handler.inference(
                "test-provider", request_model="test-model"
            )
            invocation.suspend()
        invocation.input_messages = _user_message("streamed")
        invocation.stop()

        self.assertEqual(self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [True])

    def test_nested_override_restores_outer_mode(self) -> None:
        handler = self._handler("NO_CONTENT")
        with override_content_capturing_mode(ContentCapturingMode.SPAN_ONLY):
            with override_content_capturing_mode(
                ContentCapturingMode.NO_CONTENT
            ):
                self._chat(handler, "inner")
            self._chat(handler, "outer")

        self.assertEqual(
            self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [False, True]
        )

    def test_override_is_read_from_the_parent_context(self) -> None:
        handler = self._handler("NO_CONTENT")
        with override_content_capturing_mode(ContentCapturingMode.SPAN_ONLY):
            request_context = get_current()

        with handler.inference(
            "test-provider",
            request_model="test-model",
            context=request_context,
        ) as invocation:
            invocation.input_messages = _user_message("hi")

        self.assertEqual(self._spans_with(GenAI.GEN_AI_INPUT_MESSAGES), [True])

    def test_override_applies_to_other_operations(self) -> None:
        handler = self._handler("NO_CONTENT")

        def tool() -> None:
            with handler.tool("get_weather") as invocation:
                invocation.arguments = {"city": "Paris"}

        def workflow() -> None:
            with handler.workflow(name="pipeline") as invocation:
                invocation.input_messages = _user_message("hi")

        def local_agent() -> None:
            with handler.invoke_local_agent(agent_name="helper") as invocation:
                invocation.input_messages = _user_message("hi")

        def retrieval() -> None:
            with handler.retrieval() as invocation:
                invocation.query_text = "hi"

        operations: list[tuple[Callable[[], None], str]] = [
            (tool, GenAI.GEN_AI_TOOL_CALL_ARGUMENTS),
            (workflow, GenAI.GEN_AI_INPUT_MESSAGES),
            (local_agent, GenAI.GEN_AI_INPUT_MESSAGES),
            (retrieval, GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT),
        ]
        for run, attribute in operations:
            with self.subTest(operation=run.__name__):
                self.span_exporter.clear()
                run()
                with override_content_capturing_mode(
                    ContentCapturingMode.SPAN_ONLY
                ):
                    run()

                self.assertEqual(self._spans_with(attribute), [False, True])


if __name__ == "__main__":
    unittest.main()
