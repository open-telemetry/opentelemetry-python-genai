# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import unittest

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
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=tracer_provider)

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

    def test_with_conversation_id(self):
        from opentelemetry.semconv._incubating.attributes import (
            gen_ai_attributes as GenAI,
        )

        invocation = self.handler.workflow(name="test")
        invocation.conversation_id = "conv-123"
        invocation.stop()
        spans = self.span_exporter.get_finished_spans()
        assert spans[0].attributes is not None
        assert spans[0].attributes[GenAI.GEN_AI_CONVERSATION_ID] == "conv-123"
