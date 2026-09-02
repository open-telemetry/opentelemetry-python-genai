# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import unittest

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
from opentelemetry.trace import INVALID_SPAN
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    TextPart,
)


class TestWorkflowInvocation(unittest.TestCase):
    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        self.log_exporter = InMemoryLogRecordExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
        )

    def test_default_values(self):
        invocation = self.handler.workflow(name=None)
        invocation.stop()
        assert invocation._name is None
        assert invocation._operation_name == "invoke_workflow"
        assert not invocation.input_messages
        assert not invocation.output_messages
        assert invocation.span is not INVALID_SPAN
        assert not invocation.attributes

    def test_custom_name(self):
        invocation = self.handler.workflow(name="customer_support_pipeline")
        invocation.stop()
        assert invocation._name == "customer_support_pipeline"

    def test_with_input_messages(self):
        msg = InputMessage(role="user", parts=[TextPart(content="hello")])
        invocation = self.handler.workflow(name="test")
        invocation.input_messages = [msg]
        invocation.stop()
        assert len(invocation.input_messages) == 1
        assert invocation.input_messages[0].role == "user"

    def test_with_output_messages(self):
        msg = OutputMessage(
            role="assistant",
            parts=[TextPart(content="hi")],
            finish_reason="stop",
        )
        invocation = self.handler.workflow(name="test")
        invocation.output_messages = [msg]
        invocation.stop()
        assert len(invocation.output_messages) == 1
        assert invocation.output_messages[0].finish_reason == "stop"

    def test_inherits_genai_invocation(self):
        invocation = self.handler.workflow(name="test")
        invocation.attributes["key"] = "value"
        invocation.stop()
        spans = self.span_exporter.get_finished_spans()
        assert spans[0].attributes is not None
        assert spans[0].attributes["key"] == "value"

    def test_default_lists_are_independent(self):
        """Ensure separate invocations get separate list instances."""
        inv1 = self.handler.workflow(name=None)
        inv2 = self.handler.workflow(name=None)
        inv1.input_messages.append(InputMessage(role="user", parts=[]))
        assert len(inv2.input_messages) == 0
        inv1.stop()
        inv2.stop()

    def test_default_attributes_are_independent(self):
        inv1 = self.handler.workflow(name=None)
        inv2 = self.handler.workflow(name=None)
        inv1.attributes["foo"] = "bar"
        assert "foo" not in inv2.attributes
        inv1.stop()
        inv2.stop()

    def test_full_construction(self):
        inp = InputMessage(role="user", parts=[TextPart(content="query")])
        out = OutputMessage(
            role="assistant",
            parts=[TextPart(content="answer")],
            finish_reason="stop",
        )
        invocation = self.handler.workflow(name="my_workflow")
        invocation.input_messages = [inp]
        invocation.output_messages = [out]
        invocation.stop()
        assert invocation._name == "my_workflow"
        assert len(invocation.input_messages) == 1
        assert len(invocation.output_messages) == 1
        assert invocation.output_messages[0].parts[0].content == "answer"

    def test_emit_event_uses_invocation_context(self):
        invocation = self.handler.workflow(name="durable_workflow")
        invocation.emit_event(
            "gen_ai.agent.checkpointed",
            {"gen_ai.agent.checkpoint.id": "ckpt-1"},
            body="Agent execution checkpointed",
        )
        invocation.stop()

        records = self.log_exporter.get_finished_logs()
        assert len(records) == 1
        record = records[0].log_record
        assert record.event_name == "gen_ai.agent.checkpointed"
        assert record.body == "Agent execution checkpointed"
        assert record.attributes is not None
        assert record.attributes["gen_ai.agent.checkpoint.id"] == "ckpt-1"
        assert record.trace_id == invocation.span.get_span_context().trace_id
        assert record.span_id == invocation.span.get_span_context().span_id

    def test_emit_event_after_stop_is_ignored(self):
        invocation = self.handler.workflow(name="durable_workflow")
        invocation.stop()
        invocation.emit_event("test.event", {})

        assert self.log_exporter.get_finished_logs() == ()


class TestEmitEventMatchesPR507(unittest.TestCase):
    """`emit_event` is a copy of PR #507's hunk, not a competing API."""

    def test_local_emit_event_matches_the_stored_pr507_source(self):
        from pathlib import Path

        tests_dir = Path(__file__).resolve().parent
        stored = tests_dir / "fixtures" / "pr507_emit_event.py.txt"
        if not stored.is_file():
            self.skipTest("PR #507 reference source is not packaged")

        source = (
            tests_dir.parent / "src/opentelemetry/util/genai/_invocation.py"
        ).read_text()
        for fragment in stored.read_text().split("\n\n", 1):
            assert fragment.strip("\n") in source
