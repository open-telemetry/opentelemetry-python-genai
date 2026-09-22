# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from ..common.base import TestCase as CommonTestCaseBase
from .base import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    InteractionsResource,
)
from .util import (
    FakeAsyncStream,
    FakeStream,
    create_mock_content_event,
    create_mock_error_event,
    create_mock_fetched_interaction,
    create_mock_sse_completed_event,
    create_mock_step_event,
    parse_sse_events,
)

_SPAN_NAME = "fetch_response"


class TestCase(CommonTestCaseBase):
    """Shared assertions for ``interactions.get`` (fetch response)."""

    def setUp(self) -> None:
        super().setUp()
        if not _HAS_INTERACTIONS:
            raise unittest.SkipTest(
                "Interactions are not supported in this version of google-genai"
            )
        if self.__class__ == TestCase:
            raise unittest.SkipTest("Skipping testcase base.")
        self._original_get = InteractionsResource.get
        self._original_async_get = AsyncInteractionsResource.get
        self._installed = False
        self._interaction: Any = None
        self._exception: BaseException | None = None
        self._stream_error: BaseException | None = None
        self._stream_close_error: BaseException | None = None
        self._malformed_events = False
        self._error_event = False
        self._delta_events = False
        self._resumed_delta_events = False
        self._tool_call_events = False
        self._server_tool_events = False
        self._real_model_events = False
        self._start_content_events = False
        self._user_input_events = False
        self._last_stream: Any = None
        self._last_call: dict[str, Any] | None = None

    def tearDown(self) -> None:
        super().tearDown()
        InteractionsResource.get = self._original_get
        AsyncInteractionsResource.get = self._original_async_get

    # -- configuration ---------------------------------------------------

    def configure_valid_fetch(self, **kwargs: Any) -> None:
        self._install()
        self._interaction = create_mock_fetched_interaction(**kwargs)

    def configure_exception(self, e: BaseException) -> None:
        self._install()
        self._exception = e

    def configure_stream_error(self, e: BaseException) -> None:
        self._install()
        self._stream_error = e

    def configure_stream_close_error(self, e: BaseException) -> None:
        self._install()
        self._stream_close_error = e

    def configure_error_event(self) -> None:
        self._install()
        self._error_event = True

    def configure_delta_events(self) -> None:
        """Stream content as step deltas, with no steps on the interaction.

        This is what the API really does for a streamed fetch: the completion
        event carries metadata only.
        """
        self._install()
        self._delta_events = True

    def configure_tool_call_events(self) -> None:
        """Stream a function call and a text answer, with no steps on the
        interaction -- what a streamed fetch of a tool-using interaction looks
        like."""
        self._install()
        self._tool_call_events = True

    def configure_server_tool_events(self) -> None:
        """Stream a server tool call whose payload arrives structured.

        A server tool's arguments and results come as models on the delta, not
        as JSON string fragments the way a function call's do.
        """
        self._install()
        self._server_tool_events = True

    def configure_real_model_events(self) -> None:
        """Stream events parsed into the SDK's own models."""
        self._install()
        self._real_model_events = True

    def configure_start_content_events(self) -> None:
        """Stream model output whose text starts on the step itself.

        One step continues into a delta, the other has no delta at all.
        """
        self._install()
        self._start_content_events = True

    def configure_user_input_events(self) -> None:
        """Stream a user_input step ahead of the model output.

        A fetched interaction carries the prompt as a step of its own; it must
        not be reported as assistant output.
        """
        self._install()
        self._user_input_events = True

    def configure_resumed_delta_events(self) -> None:
        """Deltas for a step whose step.start went to the dropped connection.

        What a resumed stream looks like when the cursor lands mid-step.
        """
        self._install()
        self._resumed_delta_events = True

    def configure_malformed_events(self) -> None:
        self._install()
        self._malformed_events = True

    def _install(self) -> None:
        if self._installed:
            return
        self._installed = True
        self.reset_client()
        self.reset_instrumentation()

        def _result(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            self._last_call = {"args": args, "kwargs": kwargs}
            if self._exception is not None:
                raise self._exception
            interaction = self._interaction
            if interaction is None:
                interaction = create_mock_fetched_interaction()
            if not kwargs.get("stream"):
                return interaction
            if self._malformed_events:
                return [object(), {"event_type": None}]
            if self._tool_call_events:
                interaction.steps = None
                interaction.output_text = None
                return [
                    create_mock_step_event(
                        "step.start",
                        0,
                        step={
                            "type": "function_call",
                            "id": "call-1",
                            "name": "get_weather",
                        },
                    ),
                    create_mock_step_event(
                        "step.delta",
                        0,
                        delta={
                            "type": "arguments_delta",
                            "arguments": '{"city":',
                        },
                    ),
                    create_mock_step_event(
                        "step.delta",
                        0,
                        delta={
                            "type": "arguments_delta",
                            "arguments": '"Paris"}',
                        },
                    ),
                    create_mock_step_event("step.stop", 0),
                    create_mock_step_event(
                        "step.start", 1, step_type="model_output"
                    ),
                    create_mock_step_event("step.delta", 1, text="Sunny."),
                    create_mock_sse_completed_event(interaction),
                ]
            if self._user_input_events:
                interaction.steps = None
                interaction.output_text = None
                return parse_sse_events(
                    [
                        {
                            "event_type": "step.start",
                            "index": 0,
                            "step": {
                                "type": "user_input",
                                "content": [
                                    {"type": "text", "text": "What is OTel?"}
                                ],
                            },
                        },
                        {
                            "event_type": "step.start",
                            "index": 1,
                            "step": {"type": "model_output"},
                        },
                        {
                            "event_type": "step.delta",
                            "index": 1,
                            "delta": {
                                "type": "text",
                                "text": "An observability framework.",
                            },
                        },
                    ]
                ) + [create_mock_sse_completed_event(interaction)]
            if self._start_content_events:
                interaction.steps = None
                interaction.output_text = None
                return parse_sse_events(
                    [
                        {
                            "event_type": "step.start",
                            "index": 0,
                            "step": {
                                "type": "model_output",
                                "content": [{"type": "text", "text": "Hello"}],
                            },
                        },
                        {
                            "event_type": "step.delta",
                            "index": 0,
                            "delta": {"type": "text", "text": " world"},
                        },
                        {
                            "event_type": "step.start",
                            "index": 1,
                            "step": {
                                "type": "model_output",
                                "content": [
                                    {"type": "text", "text": "No deltas."}
                                ],
                            },
                        },
                    ]
                ) + [create_mock_sse_completed_event(interaction)]
            if self._real_model_events:
                interaction.steps = None
                interaction.output_text = None
                return parse_sse_events(
                    [
                        {
                            "event_type": "step.start",
                            "index": 0,
                            "step": {
                                "type": "google_search_call",
                                "id": "s-1",
                                "arguments": {"queries": ["otel"]},
                            },
                        },
                        {
                            "event_type": "step.start",
                            "index": 1,
                            "step": {
                                "type": "google_search_result",
                                "call_id": "s-1",
                                "result": [],
                            },
                        },
                        {
                            "event_type": "step.delta",
                            "index": 1,
                            "delta": {
                                "type": "google_search_result",
                                # A list of SDK models, not plain dicts.
                                "result": [
                                    {
                                        "title": "OpenTelemetry",
                                        "url": "https://opentelemetry.io",
                                    }
                                ],
                                "is_error": False,
                            },
                        },
                    ]
                ) + [create_mock_sse_completed_event(interaction)]
            if self._server_tool_events:
                interaction.steps = None
                interaction.output_text = None
                return [
                    create_mock_step_event(
                        "step.start",
                        0,
                        step={
                            "type": "code_execution_call",
                            "id": "code-1",
                            "arguments": {},
                        },
                    ),
                    create_mock_step_event(
                        "step.delta",
                        0,
                        delta={
                            "type": "code_execution_call",
                            "arguments": {
                                "code": "print(1)",
                                "language": "PYTHON",
                            },
                        },
                    ),
                    create_mock_step_event(
                        "step.start",
                        1,
                        step={
                            "type": "code_execution_result",
                            "call_id": "code-1",
                        },
                    ),
                    create_mock_step_event(
                        "step.delta",
                        1,
                        delta={
                            "type": "code_execution_result",
                            "result": "1",
                            "is_error": False,
                        },
                    ),
                    create_mock_sse_completed_event(interaction),
                ]
            if self._resumed_delta_events:
                interaction.steps = None
                interaction.output_text = None
                return [
                    create_mock_step_event("step.stop", 1),
                    create_mock_step_event("step.delta", 1, text="resumed "),
                    create_mock_step_event("step.delta", 1, text="text."),
                    create_mock_sse_completed_event(interaction),
                ]
            if self._delta_events:
                interaction.steps = None
                interaction.output_text = None
                return [
                    create_mock_step_event(
                        "step.start", 0, step_type="thought"
                    ),
                    create_mock_step_event("step.delta", 0, text=None),
                    create_mock_step_event(
                        "step.start", 1, step_type="model_output"
                    ),
                    create_mock_step_event("step.delta", 1, text="Hello "),
                    create_mock_step_event("step.delta", 1, text="there."),
                    create_mock_step_event("step.stop", 1),
                    create_mock_sse_completed_event(interaction),
                ]
            if self._error_event:
                return [create_mock_content_event(), create_mock_error_event()]
            return [
                create_mock_content_event(),
                create_mock_sse_completed_event(interaction),
            ]

        def _sync_get(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = _result(args, kwargs)
            if isinstance(result, list):
                self._last_stream = FakeStream(
                    result, self._stream_error, self._stream_close_error
                )
                return self._last_stream
            return result

        async def _async_get(_self: Any, *args: Any, **kwargs: Any) -> Any:
            result = _result(args, kwargs)
            if isinstance(result, list):
                self._last_stream = FakeAsyncStream(
                    result, self._stream_error, self._stream_close_error
                )
                return self._last_stream
            return result

        InteractionsResource.get = _sync_get
        AsyncInteractionsResource.get = _async_get

    # -- abstract --------------------------------------------------------

    def run_fetch(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()

    def run_streaming_fetch(self, *args: Any, **kwargs: Any) -> list[Any]:
        raise NotImplementedError()

    def run_streaming_fetch_with_caller_error(
        self, *args: Any, **kwargs: Any
    ) -> None:
        raise NotImplementedError()

    def run_streaming_fetch_closed_early(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        raise NotImplementedError()

    def drain_stream(self, *args: Any, **kwargs: Any) -> Any:
        """Drain a stream and return the wrapper itself."""
        raise NotImplementedError()

    # -- tests -----------------------------------------------------------

    def test_instrumentation_does_not_break_core_functionality(self) -> None:
        self.configure_valid_fetch(
            interaction_id="abc-123", output_text="Yep, it works!"
        )
        response = self.run_fetch("abc-123")
        self.assertEqual(response.id, "abc-123")
        self.assertEqual(response.output_text, "Yep, it works!")

    def test_positional_id_recorded_on_span(self) -> None:
        self.configure_valid_fetch(interaction_id="abc-123")
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.response.id"], "abc-123")

    def test_keyword_id_recorded_on_span(self) -> None:
        self.configure_valid_fetch(interaction_id="kw-456")
        self.run_fetch(id="kw-456")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.response.id"], "kw-456")

    def test_generated_span_has_minimal_genai_attributes(self) -> None:
        self.configure_valid_fetch(interaction_id="abc-123")
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["gen_ai.operation.name"], "fetch_response"
        )
        self.assertEqual(span.attributes["gen_ai.provider.name"], "gemini")
        self.assertEqual(
            span.attributes["server.address"],
            "generativelanguage.googleapis.com",
        )
        self.assertIsInstance(span.attributes["gen_ai.response.id"], str)

    def test_generated_span_has_vertex_ai_system_when_configured(self) -> None:
        self.set_use_vertex(True)
        self.configure_valid_fetch()
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.provider.name"], "vertex_ai")

    def test_response_model_and_status_recorded(self) -> None:
        self.configure_valid_fetch(model_name="gemini-2.5-pro")
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["gen_ai.response.model"], "gemini-2.5-pro"
        )
        self.assertEqual(
            span.attributes["gen_ai.response.status"], "completed"
        )
        self.assertEqual(
            span.attributes["gen_ai.response.finish_reasons"], ("stop",)
        )

    def test_failed_status_is_not_a_fetch_error(self) -> None:
        self.configure_valid_fetch(status="failed")
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.response.status"], "failed")
        self.assertEqual(
            span.attributes["gen_ai.response.finish_reasons"], ("error",)
        )
        self.assertNotIn("error.type", span.attributes)

    def test_in_progress_status_has_no_finish_reason(self) -> None:
        self.configure_valid_fetch(status="in_progress")
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["gen_ai.response.status"], "in_progress"
        )
        self.assertNotIn("gen_ai.response.finish_reasons", span.attributes)
        self.assertNotIn("error.type", span.attributes)

    def test_status_values_map_onto_semconv_value_set(self) -> None:
        # gen_ai.response.status has a well-known value set, so a provider
        # status is reported as the closest member, or as a provider-specific
        # value when no member applies.
        cases = (
            ("in_progress", "in_progress", None),
            ("completed", "completed", ("stop",)),
            ("incomplete", "incomplete", ("length",)),
            ("failed", "failed", ("error",)),
            ("cancelled", "cancelled", ("error",)),
            # No semconv member covers a spent budget; "incomplete" is the
            # closest: generation stopped before completing.
            ("budget_exceeded", "incomplete", ("length",)),
            # No member covers waiting on the caller, so this one passes
            # through as a provider-specific value.
            ("requires_action", "requires_action", ("tool_calls",)),
        )
        for status, _, _ in cases:
            self.configure_valid_fetch(status=status)
            self.run_fetch("abc-123")

        spans = [
            span
            for span in self.otel.get_finished_spans()
            if span.name == _SPAN_NAME
        ]
        self.assertEqual(len(spans), len(cases))
        for span, (status, expected_status, expected_reasons) in zip(
            spans, cases
        ):
            with self.subTest(status=status):
                self.assertEqual(
                    span.attributes["gen_ai.response.status"], expected_status
                )
                self.assertEqual(
                    span.attributes.get("gen_ai.response.finish_reasons"),
                    expected_reasons,
                )

    def test_no_token_usage_recorded(self) -> None:
        self.configure_valid_fetch(input_tokens=15, output_tokens=25)
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        for attribute in span.attributes:
            self.assertFalse(
                attribute.startswith("gen_ai.usage."),
                f"fetch_response must not report token usage, got {attribute}",
            )
        self.assertEqual(
            self.otel.get_metrics_data_named("gen_ai.client.token.usage"), []
        )

    def test_records_duration_metric(self) -> None:
        self.configure_valid_fetch()
        self.run_fetch("abc-123")
        self.otel.assert_has_metrics_data_named(
            "gen_ai.client.operation.duration"
        )

    def test_stream_attribute_absent_when_not_streaming(self) -> None:
        self.configure_valid_fetch()
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertNotIn("gen_ai.request.stream", span.attributes)
        self.assertNotIn("gen_ai.request.stream_cursor", span.attributes)

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_span_attributes_with_content_capture(self) -> None:
        self.configure_valid_fetch(
            output_text="Fetched response!",
            system_instruction="Be terse.",
        )
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
            '[{"role":"assistant","parts":[{"content":"Fetched response!","type":"text"}],"finish_reason":null,"name":null}]',
        )
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS],
            '[{"content":"Be terse.","type":"text"}]',
        )
        # A fetched response does not carry the original request's input.
        self.assertNotIn(
            GenAIAttributes.GEN_AI_INPUT_MESSAGES, span.attributes
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"},
    )
    def test_span_attributes_no_content_capture(self) -> None:
        self.configure_valid_fetch(
            output_text="Fetched response!",
            system_instruction="Be terse.",
        )
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        for attribute in (
            GenAIAttributes.GEN_AI_INPUT_MESSAGES,
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES,
            GenAIAttributes.GEN_AI_SYSTEM_INSTRUCTIONS,
        ):
            self.assertNotIn(attribute, span.attributes)

    def test_tool_definitions_recorded(self) -> None:
        self.configure_valid_fetch(
            tools=[
                {
                    "type": "function",
                    "name": "dict_tool",
                    "description": "Dict tool desc",
                }
            ]
        )
        self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["gen_ai.tool.definitions"],
            '[{"name":"dict_tool","description":"Dict tool desc","parameters":null,"type":"function"}]',
        )

    def test_fetch_exception_records_error(self) -> None:
        self.configure_exception(ValueError("Uh oh!"))
        with self.assertRaises(ValueError):
            self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "ValueError")

    def test_cancelled_fetch_records_error(self) -> None:
        self.configure_exception(asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            self.run_fetch("abc-123")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["error.type"], "asyncio.exceptions.CancelledError"
        )

    def test_blank_id_is_not_instrumented(self) -> None:
        self.configure_valid_fetch()
        self.run_fetch("")
        self.otel.assert_does_not_have_span_named(_SPAN_NAME)

    # -- streaming -------------------------------------------------------

    def test_streaming_fetch_generates_span(self) -> None:
        self.configure_valid_fetch(interaction_id="stream-id-1")
        events = self.run_streaming_fetch("stream-id-1", stream=True)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1].interaction.id, "stream-id-1")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.response.id"], "stream-id-1")
        self.assertEqual(
            span.attributes["gen_ai.response.status"], "completed"
        )
        # fetch_response has no gen_ai.request.stream attribute; only a resumed
        # fetch is distinguished, by its cursor.
        self.assertNotIn("gen_ai.request.stream", span.attributes)
        self.assertNotIn("gen_ai.request.stream_cursor", span.attributes)

    def test_resumed_streaming_fetch_records_cursor(self) -> None:
        self.configure_valid_fetch(interaction_id="stream-id-2")
        self.run_streaming_fetch(
            "stream-id-2", stream=True, last_event_id="event-42"
        )
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes["gen_ai.request.stream_cursor"], "event-42"
        )
        self.assertIsInstance(
            span.attributes["gen_ai.request.stream_cursor"], str
        )

    def test_last_event_id_ignored_when_not_streaming(self) -> None:
        self.configure_valid_fetch()
        self.run_fetch("abc-123", last_event_id="event-42")
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertNotIn("gen_ai.request.stream_cursor", span.attributes)

    def test_streaming_fetch_records_no_token_usage(self) -> None:
        self.configure_valid_fetch(input_tokens=15, output_tokens=25)
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        for attribute in span.attributes:
            self.assertFalse(attribute.startswith("gen_ai.usage."))

    def test_streaming_request_exception_records_error(self) -> None:
        self.configure_exception(ValueError("stream request failed"))
        with self.assertRaises(ValueError):
            self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "ValueError")

    def test_mid_stream_error_records_error(self) -> None:
        self.configure_valid_fetch()
        self.configure_stream_error(ConnectionError("boom"))
        with self.assertRaises(ConnectionError):
            self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "ConnectionError")

    def test_caller_side_error_records_error_and_closes_stream(self) -> None:
        self.configure_valid_fetch()
        with self.assertRaises(RuntimeError):
            self.run_streaming_fetch_with_caller_error("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "RuntimeError")
        self.assertTrue(self._last_stream.closed)

    def test_close_error_records_error_and_propagates(self) -> None:
        self.configure_valid_fetch()
        self.configure_stream_close_error(OSError("close failed"))
        with self.assertRaises(OSError):
            self.run_streaming_fetch_closed_early("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "OSError")

    def test_malformed_stream_event_does_not_raise(self) -> None:
        # _process_chunk runs outside the wrapper's try/except, so a raise there
        # would both break the SDK contract and leave the span open.
        self.configure_valid_fetch()
        self.configure_malformed_events()
        events = self.run_streaming_fetch("abc-123", stream=True)
        self.assertEqual(len(events), 2)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertNotIn("error.type", span.attributes)
        # No completion event was seen, so nothing beyond the request is known.
        self.assertNotIn("gen_ai.response.status", span.attributes)

    def test_in_band_stream_error_fails_the_span(self) -> None:
        # The provider reports a generation failure as an `error` event over a
        # successful HTTP response, without raising.
        self.configure_valid_fetch()
        self.configure_error_event()
        events = self.run_streaming_fetch("abc-123", stream=True)
        self.assertEqual(len(events), 2)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["error.type"], "service_unavailable")
        # No completion event arrived, so nothing is known about the response.
        self.assertNotIn("gen_ai.response.status", span.attributes)

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_content_captured_from_step_deltas(self) -> None:
        self.configure_valid_fetch()
        self.configure_delta_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
            '[{"role":"assistant","parts":[{"content":"Hello there.","type":"text"}],"finish_reason":null,"name":null}]',
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT"},
    )
    def test_streaming_step_deltas_not_captured_without_content(self) -> None:
        self.configure_valid_fetch()
        self.configure_delta_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertNotIn(
            GenAIAttributes.GEN_AI_OUTPUT_MESSAGES, span.attributes
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_resumed_stream_captures_deltas_without_step_start(self) -> None:
        self.configure_valid_fetch()
        self.configure_resumed_delta_events()
        self.run_streaming_fetch(
            "abc-123", stream=True, last_event_id="event-7"
        )
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES],
            '[{"role":"assistant","parts":[{"content":"resumed text.","type":"text"}],"finish_reason":null,"name":null}]',
        )

    @patch.dict(
        "os.environ",
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
            # A configured completion hook legitimately turns capture back on,
            # since the hook consumes the content.
            "OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK": "",
        },
    )
    def test_no_text_buffered_without_content_capture(self) -> None:
        # Buffering the whole response would defeat incremental consumption, and
        # it is only observable on the wrapper itself.
        self.configure_valid_fetch()
        self.configure_delta_events()
        wrapper = self.drain_stream("abc-123", stream=True)
        self.assertFalse(wrapper._self_capture_content)
        self.assertEqual(wrapper._self_content.parts(), [])
        self.assertEqual(wrapper._self_content._steps, {})

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_tool_call_captured_from_step_events(self) -> None:
        self.configure_valid_fetch()
        self.configure_tool_call_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        messages = json.loads(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        self.assertEqual(
            messages[0]["parts"],
            [
                {
                    "id": "call-1",
                    "name": "get_weather",
                    # Streamed argument fragments are reassembled and parsed.
                    "arguments": {"city": "Paris"},
                    "type": "tool_call",
                },
                {"content": "Sunny.", "type": "text"},
            ],
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_server_tool_captured_from_step_events(self) -> None:
        self.configure_valid_fetch()
        self.configure_server_tool_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        messages = json.loads(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        self.assertEqual(
            messages[0]["parts"],
            [
                {
                    "id": "code-1",
                    "name": "code_execution",
                    "server_tool_call": {
                        # Merged from the structured delta, not dropped.
                        "arguments": {
                            "code": "print(1)",
                            "language": "PYTHON",
                        },
                        "type": "code_execution",
                    },
                    "type": "server_tool_call",
                },
                {
                    "id": "code-1",
                    "server_tool_call_response": {
                        "result": "1",
                        "is_error": False,
                        "type": "code_execution",
                    },
                    "type": "server_tool_call_response",
                },
            ],
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_real_models_are_serializable(self) -> None:
        # Models nested in a delta's result list must be converted before they
        # reach the span, or serializing it raises inside the caller's own call.
        self.configure_valid_fetch()
        self.configure_real_model_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        messages = json.loads(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        self.assertEqual(
            messages[0]["parts"],
            [
                {
                    "id": "s-1",
                    "name": "google_search",
                    "server_tool_call": {
                        "arguments": {"queries": ["otel"]},
                        "type": "google_search",
                    },
                    "type": "server_tool_call",
                },
                {
                    "id": "s-1",
                    "server_tool_call_response": {
                        "result": [
                            {
                                "title": "OpenTelemetry",
                                "url": "https://opentelemetry.io",
                            }
                        ],
                        "is_error": False,
                        "type": "google_search",
                    },
                    "type": "server_tool_call_response",
                },
            ],
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_captures_step_start_content(self) -> None:
        self.configure_valid_fetch()
        self.configure_start_content_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        messages = json.loads(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        self.assertEqual(
            messages[0]["parts"],
            [
                # The step's own content, continued by its delta.
                {"content": "Hello world", "type": "text"},
                # A step whose text arrived entirely on step.start.
                {"content": "No deltas.", "type": "text"},
            ],
        )

    @patch.dict(
        "os.environ",
        {"OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"},
    )
    def test_streaming_user_input_not_reported_as_output(self) -> None:
        self.configure_valid_fetch()
        self.configure_user_input_events()
        self.run_streaming_fetch("abc-123", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        messages = json.loads(
            span.attributes[GenAIAttributes.GEN_AI_OUTPUT_MESSAGES]
        )
        self.assertEqual(
            messages[0]["parts"],
            [{"content": "An observability framework.", "type": "text"}],
        )
        # A fetched response never reports the original request's input.
        self.assertNotIn(
            GenAIAttributes.GEN_AI_INPUT_MESSAGES, span.attributes
        )

    def test_early_close_finalizes_span(self) -> None:
        self.configure_valid_fetch(interaction_id="closed-early")
        self.run_streaming_fetch_closed_early("closed-early", stream=True)
        span = self.otel.get_span_named(_SPAN_NAME)
        self.assertEqual(span.attributes["gen_ai.response.id"], "closed-early")
        self.assertNotIn("error.type", span.attributes)
        self.assertTrue(self._last_stream.closed)
