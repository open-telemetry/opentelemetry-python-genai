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
from opentelemetry.util.genai.conversation_context import (
    get_ambient_conversation_id,
    with_conversation_id,
)
from opentelemetry.util.genai.handler import TelemetryHandler


class TestConversationContextPrimitives(unittest.TestCase):
    def test_get_returns_none_when_nothing_attached(self):
        assert get_ambient_conversation_id() is None

    def test_with_conversation_id_publishes_value(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            assert get_ambient_conversation_id() == "thread-1"
        finally:
            detach(token)
        assert get_ambient_conversation_id() is None

    def test_nested_scope_overrides_and_restores(self):
        outer = attach(with_conversation_id("thread-outer"))
        try:
            inner = attach(with_conversation_id("thread-inner"))
            try:
                assert get_ambient_conversation_id() == "thread-inner"
            finally:
                detach(inner)
            assert get_ambient_conversation_id() == "thread-outer"
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
        assert len(spans) == 1, (
            f"expected exactly one {name!r} span, "
            f"got {[s.name for s in self.span_exporter.get_finished_spans()]}"
        )
        return spans[0]

    def test_inference_inherits_ambient_conversation_id(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ):
                pass
        finally:
            detach(token)

        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        assert chat_span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"

    def test_explicit_conversation_id_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            ) as inference:
                inference.conversation_id = "thread-explicit"
        finally:
            detach(token)

        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        assert (
            chat_span.attributes[GenAI.GEN_AI_CONVERSATION_ID]
            == "thread-explicit"
        )

    def test_no_ambient_and_no_explicit_omits_attribute(self):
        with self.handler.inference(
            provider="openai", request_model="gpt-4o-mini"
        ):
            pass
        chat_span = self._finished_span_by_name("chat gpt-4o-mini")
        assert GenAI.GEN_AI_CONVERSATION_ID not in chat_span.attributes

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
        assert len(spans) == 2
        assert {s.attributes[GenAI.GEN_AI_CONVERSATION_ID] for s in spans} == {
            "thread-1",
            "thread-2",
        }

    def test_local_agent_inherits_ambient_conversation_id(self):
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.invoke_local_agent(agent_name="triage"):
                pass
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_agent triage")
        assert span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"

    def test_local_agent_explicit_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.invoke_local_agent(agent_name="triage") as agent:
                agent.conversation_id = "thread-explicit"
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_agent triage")
        assert (
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-explicit"
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
        assert span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"

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
        assert span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"

    def test_workflow_explicit_wins_over_ambient(self):
        token = attach(with_conversation_id("thread-ambient"))
        try:
            with self.handler.workflow(name="wf") as workflow:
                workflow.conversation_id = "thread-explicit"
        finally:
            detach(token)

        span = self._finished_span_by_name("invoke_workflow wf")
        assert (
            span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-explicit"
        )

    def test_full_nesting_shares_one_conversation_id(self):
        # The end-to-end shape the openai-agents processor produces: one
        # attach at the top, every nested span picks it up.
        token = attach(with_conversation_id("thread-1"))
        try:
            with self.handler.workflow(name="wf"):
                with self.handler.invoke_local_agent(agent_name="triage"):
                    with self.handler.inference(
                        provider="openai", request_model="gpt-4o-mini"
                    ):
                        pass
        finally:
            detach(token)

        spans = self.span_exporter.get_finished_spans()
        assert {s.name for s in spans} == {
            "invoke_workflow wf",
            "invoke_agent triage",
            "chat gpt-4o-mini",
        }
        for span in spans:
            assert (
                span.attributes[GenAI.GEN_AI_CONVERSATION_ID] == "thread-1"
            ), f"{span.name} missing the ambient conversation id"

    def test_conversation_id_stays_off_metrics(self):
        # gen_ai.conversation.id is high cardinality; metric attributes are
        # seeded from the start attributes, which must not carry it.
        token = attach(with_conversation_id("thread-1"))
        try:
            inference = self.handler.inference(
                provider="openai", request_model="gpt-4o-mini"
            )
            inference.stop()
            metric_attrs = inference._get_metric_attributes()
        finally:
            detach(token)

        assert GenAI.GEN_AI_CONVERSATION_ID not in metric_attrs
