# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from opentelemetry import context as otel_context
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
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv._incubating.attributes import user_attributes
from opentelemetry.util.genai.context_attributes import (
    set_context_scoped_attributes,
)
from opentelemetry.util.genai.handler import get_telemetry_handler


class _RecordingSampler(Sampler):
    """Samples everything, remembering the attributes it was given."""

    def __init__(self):
        self.seen = []

    def should_sample(
        self,
        parent_context,
        trace_id,
        name,
        kind=None,
        attributes=None,
        *_,
        **__,
    ):
        self.seen.append(attributes)
        return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

    def get_description(self):
        return "RecordingSampler"


@patch.dict(
    os.environ,
    {
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
        "OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true",
    },
)
class TestContextScopedAttributes(unittest.TestCase):
    def setUp(self):
        self.sampler = _RecordingSampler()
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider(sampler=self.sampler)
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self.log_exporter)
        )
        self.handler = get_telemetry_handler(
            tracer_provider=tracer_provider, logger_provider=logger_provider
        )

    def tearDown(self):
        if hasattr(get_telemetry_handler, "_default_handler"):
            delattr(get_telemetry_handler, "_default_handler")

    @contextmanager
    def _inference(self, context):
        """Run one inference invocation with `context` attached."""
        token = otel_context.attach(context)
        try:
            with self.handler.inference(
                "test-provider", request_model="test-model"
            ) as invocation:
                yield invocation
        finally:
            otel_context.detach(token)

    @property
    def span_attributes(self):
        (span,) = self.span_exporter.get_finished_spans()
        return span.attributes

    @property
    def event_attributes(self):
        (log,) = self.log_exporter.get_finished_logs()
        return log.log_record.attributes

    def test_attributes_go_only_to_the_signal_they_target(self):
        with self._inference(
            set_context_scoped_attributes(
                span_attributes={GenAI.GEN_AI_AGENT_NAME: "trip-planner"},
                log_attributes={user_attributes.USER_ID: "u-1"},
            )
        ):
            pass

        self.assertEqual(
            self.span_attributes[GenAI.GEN_AI_AGENT_NAME], "trip-planner"
        )
        self.assertNotIn(user_attributes.USER_ID, self.span_attributes)
        self.assertEqual(self.event_attributes[user_attributes.USER_ID], "u-1")
        self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, self.event_attributes)

    def test_invocation_attributes_win(self):
        context = set_context_scoped_attributes(
            span_attributes={
                GenAI.GEN_AI_PROVIDER_NAME: "from-context",
                "custom": "from-context",
            },
            log_attributes={GenAI.GEN_AI_PROVIDER_NAME: "from-context"},
        )
        with self._inference(context) as invocation:
            invocation.attributes["custom"] = "from-invocation"

        self.assertEqual(
            self.span_attributes[GenAI.GEN_AI_PROVIDER_NAME], "test-provider"
        )
        self.assertEqual(self.span_attributes["custom"], "from-invocation")
        self.assertEqual(
            self.event_attributes[GenAI.GEN_AI_PROVIDER_NAME], "test-provider"
        )

    def test_nested_contexts_merge_with_the_inner_one_winning(self):
        outer = set_context_scoped_attributes(
            span_attributes={"outer": "outer", "shared": "outer"}
        )
        with self._inference(
            set_context_scoped_attributes(
                span_attributes={"inner": "inner", "shared": "inner"},
                context=outer,
            )
        ):
            pass

        self.assertEqual(self.span_attributes["outer"], "outer")
        self.assertEqual(self.span_attributes["inner"], "inner")
        self.assertEqual(self.span_attributes["shared"], "inner")

    def test_span_attributes_are_visible_to_the_sampler(self):
        with self._inference(
            set_context_scoped_attributes(
                span_attributes={GenAI.GEN_AI_AGENT_NAME: "trip-planner"}
            )
        ):
            pass

        (seen,) = self.sampler.seen
        self.assertEqual(seen[GenAI.GEN_AI_AGENT_NAME], "trip-planner")

    def test_attributes_do_not_apply_outside_the_attached_context(self):
        with self._inference(
            set_context_scoped_attributes(
                span_attributes={GenAI.GEN_AI_AGENT_NAME: "trip-planner"}
            )
        ):
            pass
        self.span_exporter.clear()

        with self.handler.inference("test-provider"):
            pass

        self.assertNotIn(GenAI.GEN_AI_AGENT_NAME, self.span_attributes)


if __name__ == "__main__":
    unittest.main()
