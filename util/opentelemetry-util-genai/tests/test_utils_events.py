# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest
from unittest.mock import patch

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
from opentelemetry.semconv.attributes import error_attributes
from opentelemetry.util.genai._inference_invocation import (
    InferenceInvocation,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import ContentCapturingMode, Error

from .test_utils import (
    _create_input_message,
    _create_output_message,
    _create_system_instruction,
    _get_single_span,
    _get_span_attributes,
    _normalize_to_dict,
    _normalize_to_list,
)


class TestTelemetryHandlerEvents(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.tracer_provider = tracer_provider
        self.logger_provider = logger_provider
        self.telemetry_handler = TelemetryHandler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        )

    def tearDown(self):
        self.span_exporter.clear()
        self.log_exporter.clear()

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
        },
    )
    def test_emits_llm_event(self):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="event-model"
        )
        invocation.input_messages = [_create_input_message("test query")]
        invocation.system_instruction = _create_system_instruction()
        invocation.conversation_id = "event-conv-id"
        invocation.temperature = 0.7
        invocation.max_tokens = 100
        invocation.response_model_name = "response-model"
        invocation.response_id = "event-response-id"
        invocation.input_tokens = 10
        invocation.output_tokens = 20
        invocation.output_messages = [_create_output_message("test response")]
        invocation.stop()

        # Check that event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record

        # Verify event name
        self.assertEqual(
            log_record.event_name, "gen_ai.client.inference.operation.details"
        )

        # Verify event attributes
        attrs = log_record.attributes
        self.assertIsNotNone(attrs)
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "chat")
        self.assertEqual(attrs[GenAI.GEN_AI_REQUEST_MODEL], "event-model")
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "test-provider")
        self.assertEqual(attrs[GenAI.GEN_AI_CONVERSATION_ID], "event-conv-id")
        self.assertEqual(attrs[GenAI.GEN_AI_REQUEST_TEMPERATURE], 0.7)
        self.assertEqual(attrs[GenAI.GEN_AI_REQUEST_MAX_TOKENS], 100)
        self.assertEqual(attrs[GenAI.GEN_AI_RESPONSE_MODEL], "response-model")
        self.assertEqual(attrs[GenAI.GEN_AI_RESPONSE_ID], "event-response-id")
        self.assertEqual(attrs[GenAI.GEN_AI_USAGE_INPUT_TOKENS], 10)
        self.assertEqual(attrs[GenAI.GEN_AI_USAGE_OUTPUT_TOKENS], 20)

        # Verify messages are in structured format (not JSON string)
        # OpenTelemetry may convert lists to tuples, so we normalize
        input_msg = _normalize_to_dict(
            _normalize_to_list(attrs[GenAI.GEN_AI_INPUT_MESSAGES])[0]
        )
        self.assertEqual(input_msg["role"], "Human")
        self.assertEqual(
            _normalize_to_list(input_msg["parts"])[0]["content"], "test query"
        )

        output_msg = _normalize_to_dict(
            _normalize_to_list(attrs[GenAI.GEN_AI_OUTPUT_MESSAGES])[0]
        )
        self.assertEqual(output_msg["role"], "AI")
        self.assertEqual(
            _normalize_to_list(output_msg["parts"])[0]["content"],
            "test response",
        )
        self.assertEqual(output_msg["finish_reason"], "stop")

        # Verify system instruction is present in event in structured format
        sys_instr = _normalize_to_dict(
            _normalize_to_list(attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])[0]
        )
        self.assertEqual(sys_instr["content"], "You are a helpful assistant.")
        self.assertEqual(sys_instr["type"], "text")

        # Verify event context matches span context
        span = _get_single_span(self.span_exporter)
        self.assertIsNotNone(log_record.trace_id)
        self.assertIsNotNone(log_record.span_id)
        self.assertIsNotNone(span.context)
        self.assertEqual(log_record.trace_id, span.context.trace_id)
        self.assertEqual(log_record.span_id, span.context.span_id)

        # In EVENT_ONLY mode, messages must not be attached to span
        span_attrs = span.attributes or {}
        self.assertNotIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)
        self.assertNotIn(GenAI.GEN_AI_OUTPUT_MESSAGES, span_attrs)
        self.assertNotIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
        },
    )
    def test_emits_llm_event_and_span(self):
        message = _create_input_message("combined test")
        chat_generation = _create_output_message("combined response")
        system_instruction = _create_system_instruction("System prompt here")

        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="combined-model"
        )
        invocation.input_messages = [message]
        invocation.system_instruction = system_instruction
        invocation.output_messages = [chat_generation]
        invocation.stop()

        # Check span was created
        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name, "gen_ai.client.inference.operation.details"
        )
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, log_record.attributes)
        # Verify system instruction in both span and event
        self.assertIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, span_attrs)
        span_system = json.loads(span_attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS])
        self.assertEqual(span_system[0]["content"], "System prompt here")
        event_attrs = log_record.attributes
        self.assertIn(GenAI.GEN_AI_SYSTEM_INSTRUCTIONS, event_attrs)
        event_system = event_attrs[GenAI.GEN_AI_SYSTEM_INSTRUCTIONS]
        event_system_list = (
            list(event_system)
            if isinstance(event_system, tuple)
            else event_system
        )
        event_sys_instr = (
            dict(event_system_list[0])
            if isinstance(event_system_list[0], tuple)
            else event_system_list[0]
        )
        self.assertEqual(event_sys_instr["content"], "System prompt here")
        # Verify event context matches span context
        span = _get_single_span(self.span_exporter)
        self.assertIsNotNone(log_record.trace_id)
        self.assertIsNotNone(log_record.span_id)
        self.assertIsNotNone(span.context)
        self.assertEqual(log_record.trace_id, span.context.trace_id)
        self.assertEqual(log_record.span_id, span.context.span_id)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
        },
    )
    def test_emits_llm_event_with_error(self):
        class TestError(RuntimeError):
            pass

        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        message = _create_input_message("error test")
        invocation = handler.inference(
            "test-provider", request_model="error-model"
        )
        invocation.input_messages = [message]
        error = Error(message="Test error occurred", type="TestError")
        invocation.fail(error)

        # Check event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        attrs = log_record.attributes

        # Verify error attribute is present
        self.assertEqual(attrs[error_attributes.ERROR_TYPE], "TestError")
        self.assertEqual(attrs[GenAI.GEN_AI_OPERATION_NAME], "chat")
        self.assertEqual(attrs[GenAI.GEN_AI_REQUEST_MODEL], "error-model")
        # Verify event context matches span context
        span = _get_single_span(self.span_exporter)
        self.assertIsNotNone(log_record.trace_id)
        self.assertIsNotNone(log_record.span_id)
        self.assertIsNotNone(span.context)
        self.assertEqual(log_record.trace_id, span.context.trace_id)
        self.assertEqual(log_record.span_id, span.context.span_id)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "false",
        },
    )
    def test_does_not_emit_llm_event_when_emit_event_false(self):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        message = _create_input_message("emit false test")
        chat_generation = _create_output_message("emit false response")

        invocation = handler.inference(
            "test-provider", request_model="emit-false-model"
        )
        invocation.input_messages = [message]
        invocation.output_messages = [chat_generation]
        invocation.stop()

        # Check no event was emitted
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
        },
    )
    def test_does_not_emit_llm_event_by_default_for_no_content(self):
        """Test that event is not emitted by default when content_capturing is NO_CONTENT and OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT is not set."""
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="default-model"
        )
        invocation.input_messages = [_create_input_message("default test")]
        invocation.output_messages = [
            _create_output_message("default response")
        ]
        invocation.stop()

        # Check that no event was emitted (NO_CONTENT defaults to False)
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY",
        },
    )
    def test_does_not_emit_llm_event_by_default_for_span_only(self):
        """Test that event is not emitted by default when content_capturing is SPAN_ONLY and OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT is not set."""
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="default-model"
        )
        invocation.input_messages = [_create_input_message("default test")]
        invocation.output_messages = [
            _create_output_message("default response")
        ]
        invocation.stop()

        # Check that no event was emitted (SPAN_ONLY defaults to False)
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
        },
    )
    def test_emits_llm_event_by_default_for_event_only(self):
        """Test that event is emitted by default when content_capturing is EVENT_ONLY and OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT is not set."""
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="default-model"
        )
        invocation.input_messages = [_create_input_message("default test")]
        invocation.output_messages = [
            _create_output_message("default response")
        ]
        invocation.stop()

        # Check that event was emitted (EVENT_ONLY defaults to True)
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name, "gen_ai.client.inference.operation.details"
        )

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_AND_EVENT",
        },
    )
    def test_emits_llm_event_by_default_for_span_and_event(self):
        """Test that event is emitted by default when content_capturing is SPAN_AND_EVENT and OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT is not set."""
        message = _create_input_message("span and event test")
        chat_generation = _create_output_message("span and event response")
        system_instruction = _create_system_instruction("System prompt")

        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="span-and-event-model"
        )
        invocation.input_messages = [message]
        invocation.system_instruction = system_instruction
        invocation.output_messages = [chat_generation]
        invocation.stop()

        # Check span was created
        span = _get_single_span(self.span_exporter)
        span_attrs = _get_span_attributes(span)
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, span_attrs)

        # Check that event was emitted (SPAN_AND_EVENT defaults to True)
        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)
        log_record = logs[0].log_record
        self.assertEqual(
            log_record.event_name, "gen_ai.client.inference.operation.details"
        )
        self.assertIn(GenAI.GEN_AI_INPUT_MESSAGES, log_record.attributes)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
        },
    )
    def test_emit_event_determined_at_construction_time(self):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="event-model"
        )
        invocation.input_messages = [_create_input_message("test")]
        invocation.output_messages = [_create_output_message("response")]

        # Changing os.environ after construction should have no effect
        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "false"}
        ):
            invocation.stop()

        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 1)

    @patch.dict(
        os.environ,
        {
            "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "NO_CONTENT",
            "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "false",
        },
    )
    def test_emit_event_disabled_at_construction_time_not_affected_by_env_change(
        self,
    ):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="event-model"
        )
        invocation.input_messages = [_create_input_message("test")]
        invocation.output_messages = [_create_output_message("response")]

        # Changing os.environ after construction should not cause event emission
        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true"}
        ):
            invocation.stop()

        logs = self.log_exporter.get_finished_logs()
        self.assertEqual(len(logs), 0)

    def test_inference_invocation_derives_emit_event_from_content_capturing_mode(
        self,
    ):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        inv_enabled = InferenceInvocation(
            self.tracer_provider.get_tracer("test"),
            handler._meter,
            self.logger_provider.get_logger("test"),
            handler._completion_hook,
            provider="test-provider",
            content_capturing_mode=ContentCapturingMode.EVENT_ONLY,
        )
        self.assertTrue(inv_enabled._emit_event)

        inv_disabled = InferenceInvocation(
            self.tracer_provider.get_tracer("test"),
            handler._meter,
            self.logger_provider.get_logger("test"),
            handler._completion_hook,
            provider="test-provider",
            content_capturing_mode=ContentCapturingMode.NO_CONTENT,
        )
        self.assertFalse(inv_disabled._emit_event)

    def test_finish_does_not_read_env_on_hot_path(self):
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            logger_provider=self.logger_provider,
        )
        invocation = handler.inference(
            "test-provider", request_model="test-model"
        )
        invocation.output_messages = [_create_output_message()]

        with patch.object(os, "environ") as mock_environ:
            invocation.stop()
            mock_environ.get.assert_not_called()
