# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ambient ``gen_ai.conversation.id`` propagation via OTel
Context."""

from __future__ import annotations

import unittest

from opentelemetry.context import attach, detach
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai._conversation_context import (
    get_ambient_conversation_id,
    with_conversation_id,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class TestConversationContextPrimitives(unittest.TestCase):
    def test_get_returns_none_when_nothing_attached(self):
        self.assertIsNone(get_ambient_conversation_id())

    def test_with_conversation_id_publishes_value(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            self.assertEqual(get_ambient_conversation_id(), "thread-1")
        finally:
            detach(token)
        self.assertIsNone(get_ambient_conversation_id())

    def test_nested_scope_overrides_and_restores(self):
        outer = attach(with_conversation_id("thread-outer"))
        try:
            inner = attach(with_conversation_id("thread-inner"))
            try:
                self.assertEqual(get_ambient_conversation_id(), "thread-inner")
            finally:
                detach(inner)
            self.assertEqual(get_ambient_conversation_id(), "thread-outer")
        finally:
            detach(outer)


class TestAmbientFallback(unittest.TestCase):
    """Inference, invoke_agent and invoke_workflow all fall back to the
    ambient conversation id."""

    def setUp(self):
        self.span_exporter = InMemorySpanExporter()
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(tracer_provider=tracer_provider)

    def _finished_span_by_name(self, name: str):
        spans = [
            s
            for s in self.span_exporter.get_finished_spans()
            if s.name == name
        ]
        self.assertEqual(
            len(spans),
            1,
            f"expected exactly one {name!r} span, "
            f"got {[s.name for s in self.span_exporter.get_finished_spans()]}",
        )
        return spans[0]

    def test_inference_inherits_ambient_conversation_id(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ) as inference:
                self.assertEqual(inference.conversation_id, "thread-1")
        finally:
            detach(token)

        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        self.assertEqual(
            chat_span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-1"
        )

    def test_explicit_conversation_id_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.inference(
                provider="openai",
                request_model="gpt-4o-mini",
                conversation_id="thread-explicit",
            ) as inference:
                self.assertEqual(inference.conversation_id, "thread-explicit")
        finally:
            detach(token)

        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        self.assertEqual(
            chat_span.attributes[GenAI.GEN_AI_CONVERSATION_ID],
            "thread-explicit",
        )

    def test_conversation_id_set_after_construction_still_lands_on_span(self):
        # The openai instrumentation reads the Responses `conversation`
        # request parameter after the invocation exists.
        with self.handler.inference(
            provider="openai", request_model="gpt-4o-mini"
        ) as inference:
            inference.conversation_id = "thread-late"

        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        self.assertEqual(
            chat_span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-late"
        )

    def test_explicit_empty_string_does_not_fall_back_to_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.inference(
                provider="openai",
                request_model="gpt-4o-mini",
                conversation_id="",
            ) as inference:
                self.assertEqual(inference.conversation_id, "")
                # An empty id is not worth propagating, so the enclosing one
                # is left in place rather than replaced with it.
                self.assertEqual(
                    get_ambient_conversation_id(), "thread-ambient"
                )
        finally:
            detach(token)

    def test_no_ambient_and_no_explicit_omits_attribute(self):
        with self.handler.inference(
            provider="openai", request_model="gpt-4o-mini"
        ):
            pass
        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        self.assertNotIn(GenAI.GEN_AI_CONVERSATION_ID, chat_span.attributes)

    def test_sibling_scopes_do_not_leak(self):
        for cid in ("thread-1", "thread-2"):
            t = attach(with_conversation_id(cid))
            try:
                with self.handler.inference(
                    provider="openai", request_model="gpt-4o-mini"
                ):
                    pass
            finally:
                detach(t)

        spans = [
            s
            for s in self.span_exporter.get_finished_spans()
            if s.name == "chat gpt-4o-mini"
        ]
        self.assertEqual(len(spans), 2)
        self.assertEqual(
            {s.attributes[GenAI.GEN_AI_CONVERSATION_ID] for s in spans},
            {"thread-1", "thread-2"},
        )

    def test_local_agent_inherits_ambient_conversation_id(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.invoke_local_agent(agent_name="triage"):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_agent triage")
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-1"
        )

    def test_local_agent_explicit_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.invoke_local_agent(
                agent_name="triage", conversation_id="thread-explicit"
            ):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_agent triage")
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-explicit"
        )

    def test_agent_rooted_trace_propagates_to_nested_inference(self):
        # No workflow: the agent is the outermost invocation and therefore
        # the injection point.
        with self.handler.invoke_local_agent(
            agent_name="triage", conversation_id="thread-1"
        ):
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ):
                pass

        for name in ("invoke_agent triage", "chat gpt-4o-mini"):
            span = self._finished_span_by_name(name)
            self.assertEqual(
                span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-1", name
            )

    def test_remote_agent_inherits_ambient_conversation_id(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.invoke_remote_agent(
                provider="openai", agent_name="triage"
            ):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_agent triage")
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-1"
        )

    def test_workflow_inherits_ambient_conversation_id(self):
        # Covers an agents-library workflow nested inside an outer framework
        # (e.g. LangChain) that established the conversation id.
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.workflow(name="wf"):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_workflow wf")
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-1"
        )

    def test_workflow_explicit_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.workflow(
                name="wf", conversation_id="thread-explicit"
            ):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_workflow wf")
        self.assertEqual(
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID], "thread-explicit"
        )

    def test_full_nesting_shares_one_conversation_id(self):
        # The shape the openai-agents processor produces: the id is given to
        # the outermost invocation only, and every nested span picks it up.
        with self.handler.workflow(name="wf", conversation_id="thread-1"):
            with self.handler.invoke_local_agent(agent_name="triage"):
                with self.handler.inference(
                    provider="openai", request_model="gpt-4o-mini"
                ):
                    pass

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(
            {s.name for s in spans},
            {
                "invoke_workflow wf",
                "invoke_agent triage",
                "chat gpt-4o-mini",
            },
        )
        for span in spans:
            self.assertEqual(
                span.attributes[GenAI.GEN_AI_CONVERSATION_ID],
                "thread-1",
                f"{span.name} missing the ambient conversation id",
            )

    def test_conversation_id_stays_off_metrics(self):
        # gen_ai.conversation.id is high cardinality; metric attributes are
        # seeded from the start attributes, which must not carry it.
        token = attach(with_conversation_id("thread-1"))
        try:
            inference = self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            )
            self.assertEqual(inference.conversation_id, "thread-1")
            inference.stop()
            metric_attrs = inference._get_metric_attributes()
        finally:
            detach(token)

        self.assertNotIn(GenAI.GEN_AI_CONVERSATION_ID, metric_attrs)
