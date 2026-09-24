# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextvars
import logging
from unittest import TestCase
from unittest.mock import patch

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class TestInvocationFinishedInAnotherContext(TestCase):
    """An invocation started in one contextvars context and finished in another.

    This is what every async LangChain/LangGraph run does: ``on_chain_start`` and
    ``on_chain_end`` execute in different tasks, each with its own copied context.
    """

    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)

    def test_stop_in_copied_context_does_not_log_detach_error(self):
        # Start and stop each run in their own copy of the test's context, as two
        # asyncio tasks would, so the test's own context never holds the span.
        start_context = contextvars.copy_context()
        invocation = start_context.run(self.handler.workflow, name="graph")
        finish_context = contextvars.copy_context()

        with self.assertNoLogs("opentelemetry.context", level=logging.ERROR):
            finish_context.run(invocation.stop)

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(1, len(spans))
        self.assertEqual("invoke_workflow graph", spans[0].name)
        # The foreign reset is a no-op, and the test's context was never touched.
        self.assertIs(trace.get_current_span(), trace.INVALID_SPAN)


class TestInvocationWithOpaqueContextToken(TestCase):
    """A runtime context other than contextvars (``OTEL_PYTHON_CONTEXT``).

    Its tokens have no ``var`` attribute, so ``suspend`` must hand them back to
    ``opentelemetry.context.detach`` and still end the span.
    """

    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=self.tracer_provider)

    def test_stop_detaches_opaque_token_and_ends_span(self):
        opaque_token = object()
        with (
            patch(
                "opentelemetry.util.genai._invocation.attach",
                return_value=opaque_token,
            ),
            patch("opentelemetry.util.genai._invocation.detach") as detach,
        ):
            invocation = self.handler.workflow(name="graph")
            invocation.stop()

        detach.assert_called_once_with(opaque_token)
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(1, len(spans))
        self.assertEqual("invoke_workflow graph", spans[0].name)
