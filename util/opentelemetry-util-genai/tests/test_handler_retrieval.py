# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from unittest import TestCase
from unittest.mock import patch

import pytest

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.sampling import Decision, SamplingResult
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.trace import INVALID_SPAN, SpanKind
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import RetrievalInvocation
from opentelemetry.util.genai.types import Error


class _RetrievalTestBase(TestCase):
    def setUp(self) -> None:
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider = TracerProvider()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
        )

    def _get_finished_spans(self):
        return self.span_exporter.get_finished_spans()


class TelemetryHandlerRetrievalTest(_RetrievalTestBase):  # pylint: disable=too-many-public-methods
    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def test_retrieval_creates_span(self) -> None:
        invocation = self.handler.retrieval()
        self.assertIsNot(invocation.span, INVALID_SPAN)
        invocation.stop()

    def test_retrieval_span_name_with_data_source_id(self) -> None:
        invocation = self.handler.retrieval(data_source_id="H7STPQYOND")
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "retrieval H7STPQYOND")

    def test_retrieval_span_name_without_data_source_id(self) -> None:
        invocation = self.handler.retrieval()
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "retrieval")

    def test_retrieval_span_kind_is_client(self) -> None:
        invocation = self.handler.retrieval()
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].kind, SpanKind.CLIENT)

    def test_retrieval_records_monotonic_start(self) -> None:
        with patch("timeit.default_timer", return_value=42.0):
            invocation = self.handler.retrieval()
        self.assertEqual(invocation._monotonic_start_s, 42.0)
        invocation.stop()

    # ------------------------------------------------------------------
    # stop (required + conditionally required attributes)
    # ------------------------------------------------------------------

    def test_stop_sets_operation_name(self) -> None:
        invocation = self.handler.retrieval()
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(
            spans[0].attributes[GenAI.GEN_AI_OPERATION_NAME], "retrieval"
        )

    def test_stop_sets_data_source_id(self) -> None:
        invocation = self.handler.retrieval(data_source_id="DS123")
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(
            spans[0].attributes[GenAI.GEN_AI_DATA_SOURCE_ID], "DS123"
        )

    def test_stop_sets_provider_name(self) -> None:
        invocation = self.handler.retrieval(provider="pinecone")
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(
            spans[0].attributes[GenAI.GEN_AI_PROVIDER_NAME], "pinecone"
        )

    def test_stop_sets_request_model(self) -> None:
        invocation = self.handler.retrieval(
            request_model="text-embedding-ada-002"
        )
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(
            spans[0].attributes[GenAI.GEN_AI_REQUEST_MODEL],
            "text-embedding-ada-002",
        )

    def test_stop_sets_server_address_and_port(self) -> None:
        invocation = self.handler.retrieval(
            server_address="db.example.com", server_port=443
        )
        invocation.stop()

        spans = self._get_finished_spans()
        attrs = spans[0].attributes
        self.assertEqual(attrs["server.address"], "db.example.com")
        self.assertEqual(attrs["server.port"], 443)

    # ------------------------------------------------------------------
    # stop (recommended + opt-in attributes set after construction)
    # ------------------------------------------------------------------

    def test_stop_sets_top_k(self) -> None:
        invocation = self.handler.retrieval()
        invocation.top_k = 10.0
        invocation.stop()

        spans = self._get_finished_spans()
        value = spans[0].attributes[GenAI.GEN_AI_REQUEST_TOP_K]
        self.assertIsInstance(value, float)
        self.assertEqual(value, 10.0)

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        },
    )
    def test_stop_sets_query_text_when_content_capture_enabled(self) -> None:
        invocation = self.handler.retrieval()
        invocation.query_text = "What is the capital of France?"
        invocation.stop()

        spans = self._get_finished_spans()
        value = spans[0].attributes[GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT]
        self.assertIsInstance(value, str)
        self.assertEqual(value, "What is the capital of France?")

    def test_stop_suppresses_query_text_when_content_capture_disabled(
        self,
    ) -> None:
        invocation = self.handler.retrieval()
        invocation.query_text = "What is the capital of France?"
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertNotIn(
            GenAI.GEN_AI_RETRIEVAL_QUERY_TEXT, spans[0].attributes
        )

    @patch.dict(
        os.environ,
        {
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "SPAN_ONLY",
        },
    )
    def test_stop_sets_documents_when_content_capture_enabled(self) -> None:
        docs = [{"id": "doc_1", "score": 0.95}, {"id": "doc_2", "score": 0.87}]
        invocation = self.handler.retrieval()
        invocation.documents = docs
        invocation.stop()

        spans = self._get_finished_spans()
        raw = spans[0].attributes[GenAI.GEN_AI_RETRIEVAL_DOCUMENTS]
        self.assertIsInstance(raw, str)
        self.assertEqual(json.loads(raw), docs)

    def test_stop_suppresses_documents_when_content_capture_disabled(
        self,
    ) -> None:
        docs = [{"id": "doc_1", "score": 0.95}]
        invocation = self.handler.retrieval()
        invocation.documents = docs
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertNotIn(GenAI.GEN_AI_RETRIEVAL_DOCUMENTS, spans[0].attributes)

    def test_stop_sets_custom_attributes(self) -> None:
        invocation = self.handler.retrieval()
        invocation.attributes["custom.key"] = "value"
        invocation.stop()

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].attributes["custom.key"], "value")

    def test_stop_omits_none_attributes(self) -> None:
        invocation = self.handler.retrieval()
        invocation.stop()

        spans = self._get_finished_spans()
        attrs = spans[0].attributes
        self.assertNotIn(GenAI.GEN_AI_DATA_SOURCE_ID, attrs)
        self.assertNotIn(GenAI.GEN_AI_PROVIDER_NAME, attrs)
        self.assertNotIn(GenAI.GEN_AI_REQUEST_MODEL, attrs)
        self.assertNotIn(GenAI.GEN_AI_REQUEST_TOP_K, attrs)

    # ------------------------------------------------------------------
    # fail
    # ------------------------------------------------------------------

    def test_fail_sets_error_status(self) -> None:
        invocation = self.handler.retrieval()
        invocation.fail(Error(message="timeout", type=TimeoutError))

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(spans[0].status.description, "timeout")

    def test_fail_sets_error_type_attribute(self) -> None:
        invocation = self.handler.retrieval()
        invocation.fail(Error(message="bad", type=ConnectionError))

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].attributes["error.type"], "ConnectionError")

    def test_fail_sets_operation_name(self) -> None:
        invocation = self.handler.retrieval()
        invocation.fail(Error(message="err", type=RuntimeError))

        spans = self._get_finished_spans()
        self.assertEqual(
            spans[0].attributes[GenAI.GEN_AI_OPERATION_NAME], "retrieval"
        )

    def test_fail_with_exception_instance(self) -> None:
        invocation = self.handler.retrieval()
        invocation.fail(ValueError("oops"))

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(spans[0].attributes["error.type"], "ValueError")


class TelemetryHandlerRetrievalContextManagerTest(_RetrievalTestBase):
    # ------------------------------------------------------------------
    # retrieval context manager
    # ------------------------------------------------------------------

    def test_context_manager_creates_and_ends_span(self) -> None:
        with self.handler.retrieval(data_source_id="DS1") as inv:
            self.assertIsNot(inv.span, INVALID_SPAN)

        spans = self._get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].name, "retrieval DS1")

    def test_context_manager_default_invocation(self) -> None:
        with self.handler.retrieval() as inv:
            self.assertIsInstance(inv, RetrievalInvocation)
            self.assertIsNone(inv.data_source_id)
            self.assertEqual(inv._operation_name, "retrieval")

    def test_context_manager_success_has_unset_status(self) -> None:
        with self.handler.retrieval():
            pass

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].status.status_code, StatusCode.UNSET)

    def test_context_manager_reraises_exception(self) -> None:
        with pytest.raises(ValueError, match="lookup failed"):
            with self.handler.retrieval():
                raise ValueError("lookup failed")

    def test_context_manager_marks_error_on_exception(self) -> None:
        with pytest.raises(RuntimeError):
            with self.handler.retrieval():
                raise RuntimeError("store down")

        spans = self._get_finished_spans()
        self.assertEqual(spans[0].status.status_code, StatusCode.ERROR)
        self.assertEqual(spans[0].attributes["error.type"], "RuntimeError")

    def test_context_manager_sets_attributes_on_span(self) -> None:
        with self.handler.retrieval(provider="weaviate") as inv:
            inv.top_k = 5.0

        spans = self._get_finished_spans()
        attrs = spans[0].attributes
        self.assertEqual(attrs[GenAI.GEN_AI_PROVIDER_NAME], "weaviate")
        self.assertIsInstance(attrs[GenAI.GEN_AI_REQUEST_TOP_K], float)
        self.assertEqual(attrs[GenAI.GEN_AI_REQUEST_TOP_K], 5.0)


class TelemetryHandlerRetrievalSamplingTest(_RetrievalTestBase):
    def test_sampling_attributes_available_at_span_creation(self) -> None:
        """Sampling-relevant attributes must be present at start_span() time."""
        captured_attributes: dict = {}

        class AttributeCapturingSampler:  # pylint: disable=no-self-use
            def should_sample(
                self,
                parent_context,
                trace_id,
                name,
                kind=None,
                attributes=None,
                links=None,
            ):
                captured_attributes.update(attributes or {})
                return SamplingResult(Decision.RECORD_AND_SAMPLE, attributes)

            def get_description(self):
                return "AttributeCapturingSampler"

        sampler_provider = TracerProvider(sampler=AttributeCapturingSampler())
        sampler_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        handler = TelemetryHandler(tracer_provider=sampler_provider)

        invocation = handler.retrieval(
            data_source_id="DS42",
            provider="pinecone",
            server_address="db.example.com",
            server_port=443,
        )
        invocation.stop()

        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_OPERATION_NAME], "retrieval"
        )
        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_DATA_SOURCE_ID], "DS42"
        )
        self.assertEqual(
            captured_attributes[GenAI.GEN_AI_PROVIDER_NAME], "pinecone"
        )
        self.assertEqual(
            captured_attributes["server.address"], "db.example.com"
        )
        self.assertEqual(captured_attributes["server.port"], 443)
