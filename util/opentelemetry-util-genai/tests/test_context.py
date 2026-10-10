# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

from opentelemetry.context import (
    Context,
    attach,
    detach,
    get_value,
    set_value,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.semconv.attributes import (
    error_attributes,
    server_attributes,
)
from opentelemetry.test.test_base import TestBase
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.genai._conversation_context import (
    with_conversation_id,
)
from opentelemetry.util.genai._embedding_invocation import (
    SuppressedEmbeddingInvocation,
)
from opentelemetry.util.genai._fetch_response_invocation import (
    SuppressedFetchResponseInvocation,
)
from opentelemetry.util.genai._inference_invocation import (
    SuppressedInferenceInvocation,
)
from opentelemetry.util.genai._retrieval_invocation import (
    SuppressedRetrievalInvocation,
)
from opentelemetry.util.genai._tool_invocation import (
    SuppressedToolInvocation,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    CLIENT_INFERENCE_CONTEXT_KEY,
    EMBEDDING_CONTEXT_KEY,
    FETCH_RESPONSE_CONTEXT_KEY,
    RETRIEVAL_CONTEXT_KEY,
    TOOL_CONTEXT_KEY,
    EmbeddingData,
    EmbeddingInvocation,
    FetchResponseData,
    FetchResponseInvocation,
    InferenceData,
    InferenceInvocation,
    RetrievalData,
    RetrievalInvocation,
    ToolData,
    ToolInvocation,
)
from opentelemetry.util.genai.types import (
    ContentCapturingMode,
    FunctionToolDefinition,
    InputMessage,
    OutputMessage,
    RetrievalDocument,
    TextPart,
)


def get_inference_context_data(
    context: Context | None = None,
) -> InferenceData | None:
    data = get_value(CLIENT_INFERENCE_CONTEXT_KEY, context=context)
    return data if isinstance(data, InferenceData) else None


def set_inference_context_data(
    data: InferenceData, context: Context | None = None
) -> Context:
    return set_value(CLIENT_INFERENCE_CONTEXT_KEY, data, context=context)


class BaseContextTest(TestBase):
    def _harvest_metrics(self) -> dict[str, list[object]]:
        metrics = self.get_sorted_metrics()
        metrics_by_name: dict[str, list[object]] = {}
        for metric in metrics or []:
            points = getattr(metric.data, "data_points", None) or []
            metrics_by_name.setdefault(metric.name, []).extend(points)
        return metrics_by_name


class TestInferenceContext(BaseContextTest):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            CLIENT_INFERENCE_CONTEXT_KEY,
            "opentelemetry.genai.client.inference.context",
        )

    def test_get_inference_context_data_none_by_default(self) -> None:
        self.assertIsNone(get_value(CLIENT_INFERENCE_CONTEXT_KEY))

    def test_set_and_get_inference_context_data(self) -> None:
        data = InferenceData(request_model="gpt-4o")
        ctx = set_value(CLIENT_INFERENCE_CONTEXT_KEY, data)
        self.assertIs(
            get_value(CLIENT_INFERENCE_CONTEXT_KEY, context=ctx), data
        )
        self.assertIsNone(get_value(CLIENT_INFERENCE_CONTEXT_KEY))

        token = attach(ctx)
        try:
            self.assertIs(get_value(CLIENT_INFERENCE_CONTEXT_KEY), data)
        finally:
            detach(token)
        self.assertIsNone(get_value(CLIENT_INFERENCE_CONTEXT_KEY))

    def test_in_place_mutation_of_inference_context_data(self) -> None:
        data = InferenceData(usage_input_tokens=1)
        ctx = set_inference_context_data(data)
        token = attach(ctx)
        try:
            current = get_inference_context_data()
            self.assertIsNotNone(current)
            assert current is not None
            current.usage_input_tokens = 10
            current.usage_output_tokens = 20

            after = get_inference_context_data()
            assert after is not None
            self.assertEqual(after.usage_input_tokens, 10)
            self.assertEqual(after.usage_output_tokens, 20)
            self.assertIs(after, data)
        finally:
            detach(token)

    def test_inference_invocation_outer_sets_empty_object_on_context(
        self,
    ) -> None:
        self.assertIsNone(get_inference_context_data())

        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertIsInstance(invocation, InferenceInvocation)
            self.assertNotIsInstance(invocation, SuppressedInferenceInvocation)
            data = get_inference_context_data()
            self.assertIsNotNone(data)
            assert data is not None
            self.assertEqual(data, InferenceData())

            # Setting fields on outer does not mutate context object
            invocation.data.usage_input_tokens = 42
            invocation.data.usage_output_tokens = 84
            invocation.data.request_temperature = 0.7
            invocation.data.response_model = "gpt-4o-mini-2024-07-18"
            self.assertEqual(data, InferenceData())

        self.assertIsNone(get_inference_context_data())

    def test_inference_invocation_automatic_publish_on_finish(self) -> None:
        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertNotIsInstance(invocation, SuppressedInferenceInvocation)
            invocation.data.usage_input_tokens = 10
            # did not call publish_to_context()

        # Span attributes were populated automatically upon finish
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertEqual(
            spans[0].attributes.get(GenAI.GEN_AI_PROVIDER_NAME), "openai"
        )
        self.assertEqual(
            spans[0].attributes.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 10
        )

    def test_non_inference_invocations_do_not_set_inference_attributes(
        self,
    ) -> None:
        self.assertIsNone(get_inference_context_data())

        with self.handler.invoke_local_agent(agent_name="MathTutor"):
            self.assertIsNone(get_inference_context_data())

            # Nested inference invocation properly sets the inference attributes
            with self.handler.inference(
                "openai", request_model="gpt-4o-mini"
            ) as inf_inv:
                self.assertNotIsInstance(
                    inf_inv, SuppressedInferenceInvocation
                )
                self.assertIsNotNone(get_inference_context_data())

            self.assertIsNone(get_inference_context_data())

    def test_nested_inference_deduplication_and_enrichment(self) -> None:
        log_exporter = InMemoryLogRecordExporter()
        logger_provider = LoggerProvider()
        logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(log_exporter)
        )
        handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
            logger_provider=logger_provider,
        )

        with patch.dict(
            os.environ, {"OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT": "true"}
        ):
            with handler.inference(
                "openai", request_model="gpt-4o"
            ) as root_inv:
                self.assertNotIsInstance(
                    root_inv, SuppressedInferenceInvocation
                )

                with handler.inference(
                    "openai",
                    request_model="gpt-4o",
                    server_address="api.openai.com",
                    server_port=443,
                ) as nested_inv:
                    self.assertIsInstance(
                        nested_inv, SuppressedInferenceInvocation
                    )
                    self.assertIs(nested_inv.span, root_inv.span)
                    self.assertEqual(nested_inv.context, root_inv.context)
                    self.assertGreater(nested_inv._monotonic_start_s, 0.0)
                    self.assertTrue(nested_inv.span.is_recording())
                    self.assertFalse(nested_inv.should_capture_content)
                    nested_inv.data.usage_input_tokens = 15
                    nested_inv.data.usage_output_tokens = 25
                    nested_inv.data.response_model = "gpt-4o-2024-08-06"
                    nested_inv.attributes["custom.downstream"] = "enriched"
                    nested_inv.metric_attributes[
                        "custom.metric_downstream"
                    ] = "m_enriched"

                # While still in root context, context attributes contain downstream enrichment
                data = get_inference_context_data()
                self.assertIsNotNone(data)
                assert data is not None
                self.assertEqual(
                    data.server_address,
                    "api.openai.com",
                )
                self.assertEqual(data.server_port, 443)
                self.assertEqual(
                    data.response_model,
                    "gpt-4o-2024-08-06",
                )
                self.assertEqual(data.usage_input_tokens, 15)
                self.assertEqual(data.usage_output_tokens, 25)
                self.assertEqual(
                    data.attributes.get("custom.downstream"), "enriched"
                )
                self.assertEqual(
                    data.metric_attributes.get("custom.metric_downstream"),
                    "m_enriched",
                )

                # After nested exit, span is NOT ended yet (still recording)
                self.assertTrue(root_inv.span.is_recording())
                spans = self.span_exporter.get_finished_spans()
                self.assertEqual(len(spans), 0)

            # After root exit, root span is ended and event is emitted
            spans = self.span_exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            span = spans[0]
            self.assertEqual(
                span.attributes.get("gen_ai.provider.name"), "openai"
            )
            self.assertEqual(
                span.attributes.get(server_attributes.SERVER_ADDRESS),
                "api.openai.com",
            )
            self.assertEqual(
                span.attributes.get(server_attributes.SERVER_PORT), 443
            )
            self.assertEqual(
                span.attributes.get("gen_ai.response.model"),
                "gpt-4o-2024-08-06",
            )
            self.assertEqual(
                span.attributes.get("gen_ai.usage.input_tokens"), 15
            )
            self.assertEqual(
                span.attributes.get("gen_ai.usage.output_tokens"), 25
            )
            self.assertEqual(
                span.attributes.get("custom.downstream"), "enriched"
            )

            logs = log_exporter.get_finished_logs()
            self.assertEqual(len(logs), 1)
            event = logs[0].log_record
            self.assertEqual(
                event.event_name, "gen_ai.client.inference.operation.details"
            )
            self.assertIsNotNone(event.attributes)
            assert event.attributes is not None
            self.assertEqual(
                event.attributes.get(server_attributes.SERVER_ADDRESS),
                "api.openai.com",
            )
            self.assertEqual(
                event.attributes.get(server_attributes.SERVER_PORT), 443
            )
            self.assertEqual(
                event.attributes.get("gen_ai.response.model"),
                "gpt-4o-2024-08-06",
            )
            self.assertEqual(
                event.attributes.get("gen_ai.usage.input_tokens"), 15
            )
            self.assertEqual(
                event.attributes.get("gen_ai.usage.output_tokens"), 25
            )
            self.assertEqual(
                event.attributes.get("custom.downstream"), "enriched"
            )

            # Check metrics - exactly 1 operation duration point for the deduplicated invocation
            metrics = self._harvest_metrics()
            self.assertIn("gen_ai.client.operation.duration", metrics)
            duration_points = metrics["gen_ai.client.operation.duration"]
            self.assertEqual(len(duration_points), 1)
            duration_point = duration_points[0]
            self.assertEqual(
                duration_point.attributes.get(
                    server_attributes.SERVER_ADDRESS
                ),
                "api.openai.com",
            )
            self.assertEqual(
                duration_point.attributes.get(server_attributes.SERVER_PORT),
                443,
            )
            self.assertEqual(
                duration_point.attributes.get(GenAI.GEN_AI_RESPONSE_MODEL),
                "gpt-4o-2024-08-06",
            )
            self.assertEqual(
                duration_point.attributes.get("custom.metric_downstream"),
                "m_enriched",
            )
            # High-cardinality attributes must NOT leak onto metrics
            self.assertNotIn("custom.downstream", duration_point.attributes)
            self.assertNotIn(
                GenAI.GEN_AI_CONVERSATION_ID, duration_point.attributes
            )

            # Token usage metric should be enriched from context token counts
            self.assertIn("gen_ai.client.token.usage", metrics)
            token_points = metrics["gen_ai.client.token.usage"]
            token_by_type = {
                point.attributes[GenAI.GEN_AI_TOKEN_TYPE]: point
                for point in token_points
            }
            self.assertEqual(len(token_by_type), 2)
            self.assertAlmostEqual(
                token_by_type[GenAI.GenAiTokenTypeValues.INPUT.value].sum,
                15.0,
                places=3,
            )
            self.assertAlmostEqual(
                token_by_type[GenAI.GenAiTokenTypeValues.OUTPUT.value].sum,
                25.0,
                places=3,
            )
            input_token_point = token_by_type[
                GenAI.GenAiTokenTypeValues.INPUT.value
            ]
            self.assertEqual(
                input_token_point.attributes.get(
                    server_attributes.SERVER_ADDRESS
                ),
                "api.openai.com",
            )
            self.assertEqual(
                input_token_point.attributes.get(
                    server_attributes.SERVER_PORT
                ),
                443,
            )
            self.assertEqual(
                input_token_point.attributes.get(GenAI.GEN_AI_RESPONSE_MODEL),
                "gpt-4o-2024-08-06",
            )
            self.assertNotIn("custom.downstream", input_token_point.attributes)

    def test_nested_inference_invocation_does_not_end_span_on_fail(
        self,
    ) -> None:
        with self.handler.inference(
            "upstream", request_model="gpt-4o"
        ) as root_inv:
            with self.assertRaises(ValueError) as handler_error:
                with self.handler.inference(
                    "downstream", request_model="gpt-4o"
                ) as nested_inv:
                    self.assertIsInstance(
                        nested_inv, SuppressedInferenceInvocation
                    )
                    raise ValueError("downstream network failure")

            self.assertEqual(
                str(handler_error.exception), "downstream network failure"
            )
            # Root span must NOT be ended yet
            self.assertTrue(root_inv.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

            # Downstream error was caught by caller, so it is NOT recorded on context
            data = get_inference_context_data()
            self.assertIsNotNone(data)
            assert data is not None
            self.assertNotIn(error_attributes.ERROR_TYPE, data.attributes)
            self.assertNotIn(
                error_attributes.ERROR_TYPE, data.metric_attributes
            )

        # After root finishes normally, 1 span is ended without error attributes
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertNotIn(error_attributes.ERROR_TYPE, spans[0].attributes)

    def test_inner_error_caught_by_outer_does_not_fail_root(self) -> None:
        with self.handler.inference("proxy", request_model="primary-model"):
            try:
                with self.handler.inference(
                    "provider", request_model="primary-model"
                ):
                    raise RuntimeError("primary model failed")
            except RuntimeError:
                pass  # Model fallback / retry

            with self.handler.inference(
                "provider", request_model="fallback-model"
            ) as fallback_inv:
                fallback_inv.output_tokens = 20

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        root_span = spans[0]
        self.assertNotIn(error_attributes.ERROR_TYPE, root_span.attributes)
        self.assertNotEqual(root_span.status.status_code, StatusCode.ERROR)

        metrics = self._harvest_metrics()
        duration_points = metrics.get("gen_ai.client.operation.duration", [])
        self.assertEqual(len(duration_points), 1)
        self.assertNotIn(
            error_attributes.ERROR_TYPE, duration_points[0].attributes
        )

    def test_downstream_streaming_record_stream_chunk(self) -> None:
        with self.handler.inference(
            "upstream", request_model="gpt-4o"
        ) as root_inv:
            self.assertNotIsInstance(root_inv, SuppressedInferenceInvocation)
            with self.handler.inference("downstream") as nested_inv:
                self.assertIsInstance(
                    nested_inv, SuppressedInferenceInvocation
                )
                nested_inv.record_stream_chunk()
                nested_inv.record_stream_chunk()
                self.assertIsNotNone(nested_inv._ttfc_seconds)
                self.assertTrue(nested_inv._request_stream)

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        self.assertTrue(spans[0].attributes.get(GenAI.GEN_AI_REQUEST_STREAM))
        self.assertEqual(
            spans[0].attributes.get(GenAI.GEN_AI_RESPONSE_TIME_TO_FIRST_CHUNK),
            nested_inv._ttfc_seconds,
        )

        metrics = self._harvest_metrics()
        self.assertNotIn(
            "gen_ai.client.operation.time_to_first_chunk", metrics
        )
        self.assertNotIn(
            "gen_ai.client.operation.time_per_output_chunk", metrics
        )

    def test_metric_enrichment_precedence_and_error(self) -> None:
        with self.assertRaises(ValueError):
            with self.handler.inference(
                "proxy",
                request_model="gpt-4o",
                server_address="proxy.internal",
                server_port=8080,
            ) as root_inv:
                root_inv.data.usage_input_tokens = 10
                with self.handler.inference(
                    "openai",
                    request_model="gpt-4o",
                    server_address="api.openai.com",
                    server_port=443,
                ) as inner_inv:
                    inner_inv.data.usage_input_tokens = 99
                    inner_inv.data.usage_output_tokens = 50
                    inner_inv.data.response_model = "gpt-4o-2024-08-06"
                    raise ValueError("network reset")

        metrics = self._harvest_metrics()
        duration_points = metrics["gen_ai.client.operation.duration"]
        self.assertEqual(len(duration_points), 1)
        point = duration_points[0]

        # Root values take precedence over downstream context
        self.assertEqual(
            point.attributes.get(server_attributes.SERVER_ADDRESS),
            "proxy.internal",
        )
        self.assertEqual(
            point.attributes.get(server_attributes.SERVER_PORT), 8080
        )
        # Downstream fields not set on root are enriched from context
        self.assertEqual(
            point.attributes.get(GenAI.GEN_AI_RESPONSE_MODEL),
            "gpt-4o-2024-08-06",
        )
        self.assertEqual(
            point.attributes.get(error_attributes.ERROR_TYPE),
            "ValueError",
        )

        # Tokens: root input_tokens (10) takes precedence, output_tokens (50) enriched from context
        token_points = metrics["gen_ai.client.token.usage"]
        token_by_type = {
            p.attributes[GenAI.GEN_AI_TOKEN_TYPE]: p for p in token_points
        }
        self.assertAlmostEqual(
            token_by_type[GenAI.GenAiTokenTypeValues.INPUT.value].sum,
            10.0,
            places=3,
        )
        self.assertAlmostEqual(
            token_by_type[GenAI.GenAiTokenTypeValues.OUTPUT.value].sum,
            50.0,
            places=3,
        )

        # Root values take precedence over downstream context on span as well
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span_attrs = spans[0].attributes
        self.assertEqual(
            span_attrs.get(server_attributes.SERVER_ADDRESS), "proxy.internal"
        )
        self.assertEqual(span_attrs.get(server_attributes.SERVER_PORT), 8080)
        self.assertEqual(span_attrs.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 10)
        self.assertEqual(span_attrs.get(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS), 50)
        self.assertEqual(
            span_attrs.get(GenAI.GEN_AI_RESPONSE_MODEL), "gpt-4o-2024-08-06"
        )
        self.assertEqual(
            span_attrs.get(error_attributes.ERROR_TYPE), "ValueError"
        )

    def test_root_precedence_over_nested_inference(self) -> None:
        with self.handler.inference(
            "proxy-provider",
            request_model="proxy-model",
            server_address="proxy.example.com",
            server_port=8080,
        ) as root:
            root.data.request_temperature = 0.2
            root.data.usage_input_tokens = 10
            root.attributes["custom.shared"] = "root-value"

            with self.handler.inference(
                "downstream-provider",
                request_model="downstream-model",
                server_address="api.example.com",
                server_port=443,
            ) as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
                # Overwrite shared fields downstream
                inner.data.request_temperature = 0.9
                inner.data.usage_input_tokens = 100
                inner.data.usage_output_tokens = 50
                inner.data.response_id = "resp-123"
                inner.attributes["custom.shared"] = "downstream-value"
                inner.attributes["custom.downstream_only"] = "downstream-only"

            # While still in root context, context reflects downstream writes
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.request_temperature, 0.9)
            self.assertEqual(data.usage_input_tokens, 100)
            self.assertEqual(data.usage_output_tokens, 50)
            self.assertEqual(data.response_id, "resp-123")
            self.assertEqual(
                data.attributes.get("custom.downstream_only"),
                "downstream-only",
            )
            self.assertNotIn(GenAI.GEN_AI_REQUEST_TEMPERATURE, data.attributes)

        # After root finish: Option 1 root precedence applies
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        root_span = spans[0]

        # Root start attributes take precedence
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_PROVIDER_NAME),
            "proxy-provider",
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_MODEL),
            "proxy-model",
        )
        self.assertEqual(
            root_span.attributes.get(server_attributes.SERVER_ADDRESS),
            "proxy.example.com",
        )
        self.assertEqual(
            root_span.attributes.get(server_attributes.SERVER_PORT),
            8080,
        )

        # Root typed attributes take precedence
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_TEMPERATURE),
            0.2,
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS),
            10,
        )

        # Root custom attributes take precedence
        self.assertEqual(
            root_span.attributes.get("custom.shared"),
            "root-value",
        )

        # Downstream fields not set on root are enriched onto root span
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS),
            50,
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_RESPONSE_ID),
            "resp-123",
        )
        self.assertEqual(
            root_span.attributes.get("custom.downstream_only"),
            "downstream-only",
        )

    def test_multiple_inner_invocations_overwrite_context(self) -> None:
        with self.handler.inference("root-provider") as root:
            self.assertNotIsInstance(root, SuppressedInferenceInvocation)
            self.assertEqual(
                get_inference_context_data(),
                InferenceData(),
            )

            with self.handler.inference(
                "inner1-provider",
                request_model="model-1",
            ) as inner1:
                self.assertIsInstance(inner1, SuppressedInferenceInvocation)
                inner1.data.usage_input_tokens = 10
                inner1.data.usage_output_tokens = 20
                inner1.data.response_model = "resp-model-1"

            # After inner1 finishes, context has inner1's attributes
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.request_model, "model-1")
            self.assertEqual(data.usage_input_tokens, 10)
            self.assertEqual(data.response_model, "resp-model-1")

            with self.handler.inference(
                "inner2-provider",
                request_model="model-2",
            ) as inner2:
                self.assertIsInstance(inner2, SuppressedInferenceInvocation)
                inner2.data.usage_input_tokens = 30
                inner2.data.usage_output_tokens = 40
                inner2.data.response_model = "resp-model-2"

            # After inner2 finishes, inner2 overwrites inner1
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.request_model, "model-2")
            self.assertEqual(data.usage_input_tokens, 30)
            self.assertEqual(data.usage_output_tokens, 40)
            self.assertEqual(data.response_model, "resp-model-2")

        # After root finishes, root reconciles with context (inner2's values)
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        root_span = spans[0]
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_PROVIDER_NAME),
            "root-provider",
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_MODEL),
            "model-2",
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_RESPONSE_MODEL),
            "resp-model-2",
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS),
            30,
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS),
            40,
        )

    def test_nested_custom_metric_attributes_propagated(self) -> None:
        with self.handler.inference(
            "proxy", request_model="gpt-4o"
        ) as root_inv:
            root_inv.metric_attributes["root.metric"] = "root_val"
            with self.handler.inference(
                "downstream", request_model="gpt-4o"
            ) as inner_inv:
                self.assertIsInstance(inner_inv, SuppressedInferenceInvocation)
                inner_inv.metric_attributes["inner.metric"] = "inner_val"
                inner_inv.metric_attributes["root.metric"] = "inner_shadowed"

            data = get_inference_context_data()
            self.assertEqual(
                data.metric_attributes["inner.metric"], "inner_val"
            )
            self.assertEqual(
                data.metric_attributes["root.metric"], "inner_shadowed"
            )

        metrics = self._harvest_metrics()
        duration_points = metrics["gen_ai.client.operation.duration"]
        self.assertEqual(len(duration_points), 1)
        point = duration_points[0]
        # Root metric attribute takes precedence over inner metric attribute
        self.assertEqual(point.attributes.get("root.metric"), "root_val")
        # Inner metric attribute not on root is propagated
        self.assertEqual(point.attributes.get("inner.metric"), "inner_val")

    def test_suppressed_invocation_publishes_in_different_context(
        self,
    ) -> None:
        with self.handler.inference("proxy", request_model="gpt-4o"):
            inner_inv = self.handler.inference(
                "downstream", request_model="gpt-4o"
            )
            self.assertIsInstance(inner_inv, SuppressedInferenceInvocation)
            inner_inv.data.usage_input_tokens = 15
            inner_inv.data.usage_output_tokens = 25
            inner_inv.data.response_model = "gpt-4o-2024-08-06"

            # Finish inner_inv in a clean/detached context (simulating separate async task or thread)
            token = attach(Context())
            try:
                self.assertIsNone(get_inference_context_data())
                inner_inv.stop()
            finally:
                detach(token)

            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.usage_input_tokens, 15)
            self.assertEqual(data.usage_output_tokens, 25)
            self.assertEqual(data.response_model, "gpt-4o-2024-08-06")

    def test_all_non_content_attributes_on_context_and_not_in_attributes_dict(
        self,
    ) -> None:
        with self.handler.inference(
            "root-provider", request_model="root-model"
        ):
            with self.handler.inference(
                "inner-provider", request_model="inner-model"
            ) as inner:
                inner.data.request_temperature = 0.7
                inner.data.request_top_p = 0.95
                inner.data.request_top_k = 40
                inner.data.request_frequency_penalty = 0.5
                inner.data.request_presence_penalty = 0.2
                inner.data.request_max_tokens = 2048
                inner.data.request_stop_sequences = ["\n", "STOP"]
                inner.data.request_seed = 42
                inner.data.request_choice_count = 1
                inner.data.output_type = "json"
                inner.data.response_id = "resp-xyz"
                inner.data.response_finish_reasons = ["stop"]
                inner.data.usage_input_tokens = 120
                inner.data.usage_output_tokens = 60
                inner.data.usage_reasoning_output_tokens = 30
                inner.data.usage_cache_read_input_tokens = 10
                inner.data.usage_cache_write_input_tokens = 5
                inner.data.request_reasoning_level = "high"
                inner.data.prompt_name = "test-prompt"
                inner.data.prompt_version = "v1"
                inner.attributes["custom.foo"] = "bar"

            data = get_inference_context_data()
            self.assertIsNotNone(data)
            assert data is not None
            # Standard non-content attributes are on typed fields
            self.assertEqual(data.request_temperature, 0.7)
            self.assertEqual(data.request_top_p, 0.95)
            self.assertEqual(data.request_top_k, 40)
            self.assertEqual(data.request_frequency_penalty, 0.5)
            self.assertEqual(data.request_presence_penalty, 0.2)
            self.assertEqual(data.request_max_tokens, 2048)
            self.assertEqual(data.request_stop_sequences, ["\n", "STOP"])
            self.assertEqual(data.request_seed, 42)
            self.assertEqual(data.request_choice_count, 1)
            self.assertEqual(data.output_type, "json")
            self.assertEqual(data.response_id, "resp-xyz")
            self.assertEqual(data.response_finish_reasons, ["stop"])
            self.assertEqual(data.usage_input_tokens, 120)
            self.assertEqual(data.usage_output_tokens, 60)
            self.assertEqual(data.usage_reasoning_output_tokens, 30)
            self.assertEqual(data.usage_cache_read_input_tokens, 10)
            self.assertEqual(data.usage_cache_write_input_tokens, 5)
            self.assertEqual(data.request_reasoning_level, "high")
            self.assertEqual(data.prompt_name, "test-prompt")
            self.assertEqual(data.prompt_version, "v1")

            # data.attributes ONLY has custom attributes, not standard modeled fields
            self.assertEqual(data.attributes, {"custom.foo": "bar"})

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        root_span = spans[0]
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_TEMPERATURE), 0.7
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_TOP_P), 0.95
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_TOP_K), 40
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_REQUEST_MAX_TOKENS), 2048
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_RESPONSE_ID), "resp-xyz"
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 120
        )
        self.assertEqual(
            root_span.attributes.get(GenAI.GEN_AI_USAGE_OUTPUT_TOKENS), 60
        )
        self.assertEqual(root_span.attributes.get("custom.foo"), "bar")

    def test_dataclass_prototype_architecture(self) -> None:
        with self.handler.inference("openai", request_model="gpt-4o") as inv:
            # Underlying dataclass instance exists
            self.assertIsInstance(inv.data, InferenceData)

            # Dictionary references are shared
            self.assertIs(inv.attributes, inv.data.attributes)
            self.assertIs(inv.metric_attributes, inv.data.metric_attributes)

            # Context fields are on inv.data
            inv.data.usage_input_tokens = 100
            self.assertEqual(inv.data.usage_input_tokens, 100)
            inv.data.usage_output_tokens = 50
            self.assertEqual(inv.data.usage_output_tokens, 50)
            inv.data.response_model = "gpt-4o-2024-08-06"
            self.assertEqual(inv.data.response_model, "gpt-4o-2024-08-06")

            # Content access on inv.data
            from opentelemetry.util.genai.types import InputMessage, TextPart

            inv.data.input_messages.append(
                InputMessage(role="user", parts=[TextPart(content="hello")])
            )
            self.assertEqual(len(inv.data.input_messages), 1)

            # Content fields are present on inv.data
            self.assertTrue(hasattr(inv.data, "input_messages"))
            self.assertTrue(hasattr(inv.data, "output_messages"))
            self.assertTrue(hasattr(inv.data, "system_instructions"))
            self.assertTrue(hasattr(inv.data, "tool_definitions"))
            self.assertTrue(hasattr(inv.data, "prompt_variable"))

    def test_enrich_from_context_does_not_override_content(self) -> None:
        from opentelemetry.util.genai.types import (
            FunctionToolDefinition,
            InputMessage,
            OutputMessage,
            TextPart,
        )

        with self.handler.inference("openai", request_model="gpt-4o") as outer:
            outer.data.input_messages = [
                InputMessage(
                    role="user", parts=[TextPart(content="outer prompt")]
                )
            ]
            outer.data.output_messages = [
                OutputMessage(
                    role="assistant",
                    parts=[TextPart(content="outer response")],
                )
            ]
            outer.data.system_instructions = [
                TextPart(content="outer instruction")
            ]
            outer.data.prompt_variable = {"var": "outer"}
            outer.data.tool_definitions = []

            inner_data = InferenceData(
                usage_input_tokens=100,
                usage_output_tokens=50,
                input_messages=[
                    InputMessage(
                        role="user", parts=[TextPart(content="inner prompt")]
                    )
                ],
                output_messages=[
                    OutputMessage(
                        role="assistant",
                        parts=[TextPart(content="inner response")],
                    )
                ],
                system_instructions=[TextPart(content="inner instruction")],
                prompt_variable={"var": "inner"},
                tool_definitions=[
                    FunctionToolDefinition(
                        name="inner_tool", description="desc", parameters={}
                    )
                ],
            )

            outer.enrich_from_context(inner_data)

            # Non-content fields are enriched
            self.assertEqual(outer.data.usage_input_tokens, 100)
            self.assertEqual(outer.data.usage_output_tokens, 50)

            # Content fields are NOT overridden by inner
            self.assertEqual(len(outer.data.input_messages), 1)
            first_in_part = outer.data.input_messages[0].parts[0]
            self.assertIsInstance(first_in_part, TextPart)
            assert isinstance(first_in_part, TextPart)
            self.assertEqual(first_in_part.content, "outer prompt")

            self.assertEqual(len(outer.data.output_messages), 1)
            first_out_part = outer.data.output_messages[0].parts[0]
            self.assertIsInstance(first_out_part, TextPart)
            assert isinstance(first_out_part, TextPart)
            self.assertEqual(first_out_part.content, "outer response")

            self.assertEqual(outer.data.prompt_variable, {"var": "outer"})
            self.assertEqual(outer.data.tool_definitions, [])
            self.assertEqual(len(outer.data.system_instructions), 1)
            first_sys_part = outer.data.system_instructions[0]
            self.assertIsInstance(first_sys_part, TextPart)
            assert isinstance(first_sys_part, TextPart)
            self.assertEqual(first_sys_part.content, "outer instruction")

    def test_enrich_from_context_does_not_override_empty_content(self) -> None:
        from opentelemetry.util.genai.types import (
            FunctionToolDefinition,
            InputMessage,
            TextPart,
        )

        with self.handler.inference("openai", request_model="gpt-4o") as outer:
            self.assertEqual(outer.data.input_messages, [])
            self.assertIsNone(outer.data.prompt_variable)
            self.assertIsNone(outer.data.tool_definitions)

            inner_data = InferenceData(
                usage_input_tokens=100,
                input_messages=[
                    InputMessage(
                        role="user", parts=[TextPart(content="inner prompt")]
                    )
                ],
                prompt_variable={"var": "inner"},
                tool_definitions=[
                    FunctionToolDefinition(
                        name="inner_tool", description="desc", parameters={}
                    )
                ],
            )

            outer.enrich_from_context(inner_data)

            # Non-content fields are enriched
            self.assertEqual(outer.data.usage_input_tokens, 100)

            # Content fields remain default / empty
            self.assertEqual(outer.data.input_messages, [])
            self.assertIsNone(outer.data.prompt_variable)
            self.assertIsNone(outer.data.tool_definitions)

    def test_inference_context_data_merge(self) -> None:
        base = InferenceData(
            provider_name="openai",
            request_model="gpt-4o",
            request_temperature=0.5,
            usage_input_tokens=10,
            attributes={"a": 1, "shared": "base"},
            metric_attributes={"m1": "v1"},
        )
        incoming = InferenceData(
            request_model="gpt-4o-mini",
            request_temperature=0.9,
            usage_output_tokens=20,
            attributes={"b": 2, "shared": "incoming"},
            metric_attributes={"m2": "v2"},
        )

        # Merge with overwrite=True (inner publishing to context)
        dest = InferenceData()
        dest.merge(base, overwrite=True)
        dest.merge(incoming, overwrite=True)
        self.assertEqual(dest.provider_name, "openai")
        self.assertEqual(dest.request_model, "gpt-4o-mini")  # overwritten
        self.assertEqual(dest.request_temperature, 0.9)  # overwritten
        self.assertEqual(dest.usage_input_tokens, 10)
        self.assertEqual(dest.usage_output_tokens, 20)
        self.assertEqual(dest.attributes["shared"], "incoming")  # overwritten
        self.assertEqual(dest.attributes["a"], 1)
        self.assertEqual(dest.attributes["b"], 2)
        self.assertEqual(dest.metric_attributes["m1"], "v1")
        self.assertEqual(dest.metric_attributes["m2"], "v2")

        # Merge with overwrite=False (outer enriching from context)
        outer = InferenceData(
            provider_name="openai",
            request_model="gpt-4o",
            request_temperature=0.2,  # outer explicitly set temperature
            attributes={"custom": "outer", "shared": "outer"},
        )
        outer.merge(dest, overwrite=False)
        self.assertEqual(outer.request_model, "gpt-4o")  # preserved outer
        self.assertEqual(outer.request_temperature, 0.2)  # preserved outer
        self.assertEqual(outer.usage_input_tokens, 10)  # enriched from inner
        self.assertEqual(outer.usage_output_tokens, 20)  # enriched from inner
        self.assertEqual(
            outer.attributes["shared"], "outer"
        )  # preserved outer
        self.assertEqual(outer.attributes["a"], 1)  # enriched from inner

    def test_inference_invocation_property_delegation_to_data(self) -> None:
        with self.handler.inference(
            "test-provider",
            request_model="test-model",
            server_address="custom.server.com",
            server_port=42,
        ) as inv:
            # Content fields
            msg = InputMessage(role="user", parts=["hi"])
            inv.input_messages.append(msg)
            self.assertEqual(inv.data.input_messages, [msg])
            self.assertEqual(inv.input_messages, [msg])

            out_msg = OutputMessage(
                role="assistant", parts=["hello"], finish_reason="stop"
            )
            inv.output_messages = [out_msg]
            self.assertEqual(inv.data.output_messages, [out_msg])
            self.assertEqual(inv.output_messages, [out_msg])

            sys_inst = [TextPart(type="text", content="be helpful")]
            inv.system_instruction = sys_inst
            self.assertEqual(inv.data.system_instructions, sys_inst)
            self.assertEqual(inv.system_instruction, sys_inst)

            tools = [
                FunctionToolDefinition(
                    name="test_tool", description="a tool", parameters={}
                )
            ]
            inv.tool_definitions = tools
            self.assertEqual(inv.data.tool_definitions, tools)
            self.assertEqual(inv.tool_definitions, tools)

            vars_map = {"k": "v"}
            inv.prompt_variables = vars_map
            self.assertEqual(inv.data.prompt_variable, vars_map)
            self.assertEqual(inv.prompt_variables, vars_map)

            # Model and server properties
            self.assertEqual(inv.provider, "test-provider")
            self.assertEqual(inv.request_model, "test-model")
            self.assertEqual(inv.server_address, "custom.server.com")
            self.assertEqual(inv.server_port, 42)

            with self.assertRaises(AttributeError):
                setattr(inv, "provider", "other")
            with self.assertRaises(AttributeError):
                setattr(inv, "request_model", "other")
            with self.assertRaises(AttributeError):
                setattr(inv, "server_address", "other")
            with self.assertRaises(AttributeError):
                setattr(inv, "server_port", 8080)

            inv.response_model_name = "resp-model"
            self.assertEqual(inv.data.response_model, "resp-model")
            self.assertEqual(inv.response_model_name, "resp-model")

            inv.response_id = "resp-123"
            self.assertEqual(inv.data.response_id, "resp-123")
            self.assertEqual(inv.response_id, "resp-123")

            inv.finish_reasons = ["stop"]
            self.assertEqual(inv.data.response_finish_reasons, ["stop"])
            self.assertEqual(inv.finish_reasons, ["stop"])

            # Hyperparameters
            inv.temperature = 0.7
            inv.top_p = 0.95
            inv.top_k = 50
            inv.frequency_penalty = 0.1
            inv.presence_penalty = 0.2
            inv.max_tokens = 500
            inv.stop_sequences = ["\n"]
            inv.seed = 123
            inv.request_choice_count = 2
            inv.output_type = "json"
            inv.request_stream = True
            inv.ttfc_seconds = 0.15

            self.assertEqual(inv.data.request_temperature, 0.7)
            self.assertEqual(inv.data.request_top_p, 0.95)
            self.assertEqual(inv.data.request_top_k, 50)
            self.assertEqual(inv.data.request_frequency_penalty, 0.1)
            self.assertEqual(inv.data.request_presence_penalty, 0.2)
            self.assertEqual(inv.data.request_max_tokens, 500)
            self.assertEqual(inv.data.request_stop_sequences, ["\n"])
            self.assertEqual(inv.data.request_seed, 123)
            self.assertEqual(inv.data.request_choice_count, 2)
            self.assertEqual(inv.data.output_type, "json")
            self.assertTrue(inv.data.request_stream)
            self.assertEqual(inv.data.response_time_to_first_chunk, 0.15)

            # Tokens
            inv.input_tokens = 100
            inv.output_tokens = 200
            inv.thinking_tokens = 50
            inv.cache_write_input_tokens = 30
            self.assertEqual(inv.data.usage_input_tokens, 100)
            self.assertEqual(inv.data.usage_output_tokens, 200)
            self.assertEqual(inv.data.usage_reasoning_output_tokens, 50)
            self.assertEqual(inv.data.usage_cache_write_input_tokens, 30)
            self.assertEqual(inv.cache_creation_input_tokens, 30)

            inv.cache_creation_input_tokens = 40
            self.assertEqual(inv.data.usage_cache_write_input_tokens, 40)
            self.assertEqual(inv.cache_write_input_tokens, 40)

            inv.cache_read_input_tokens = 25
            self.assertEqual(inv.data.usage_cache_read_input_tokens, 25)

            # Modality tokens
            inv.text_input_tokens = 10
            inv.image_input_tokens = 20
            inv.audio_input_tokens = 30
            inv.text_output_tokens = 40
            inv.image_output_tokens = 50
            inv.audio_output_tokens = 60
            inv.text_cache_read_input_tokens = 70
            inv.image_cache_read_input_tokens = 80
            inv.audio_cache_read_input_tokens = 90

            self.assertEqual(inv.data.usage_text_input_tokens, 10)
            self.assertEqual(inv.data.usage_image_input_tokens, 20)
            self.assertEqual(inv.data.usage_audio_input_tokens, 30)
            self.assertEqual(inv.data.usage_text_output_tokens, 40)
            self.assertEqual(inv.data.usage_image_output_tokens, 50)
            self.assertEqual(inv.data.usage_audio_output_tokens, 60)
            self.assertEqual(inv.data.usage_text_cache_read_input_tokens, 70)
            self.assertEqual(inv.data.usage_image_cache_read_input_tokens, 80)
            self.assertEqual(inv.data.usage_audio_cache_read_input_tokens, 90)

            # Other fields
            inv.reasoning_level = "high"
            inv.previous_response_id = "prev-id"
            inv.conversation_compacted = True
            inv.prompt_name = "test-prompt"
            inv.prompt_version = "1.0"

            self.assertEqual(inv.data.request_reasoning_level, "high")
            self.assertEqual(inv.data.request_previous_response_id, "prev-id")
            self.assertTrue(inv.data.conversation_compacted)
            self.assertEqual(inv.data.prompt_name, "test-prompt")
            self.assertEqual(inv.data.prompt_version, "1.0")

            # Attributes and metric attributes
            inv.attributes["custom_span_attr"] = "span_val"
            self.assertEqual(
                inv.data.attributes["custom_span_attr"], "span_val"
            )

            inv.metric_attributes["custom_metric_attr"] = "metric_val"
            self.assertEqual(
                inv.data.metric_attributes["custom_metric_attr"], "metric_val"
            )

            # Stream private attribute alias
            inv._request_stream = False
            self.assertFalse(inv.request_stream)
            inv._request_stream = True
            self.assertTrue(inv.request_stream)

    def test_suppressed_inference_invocation_with_explicit_context(
        self,
    ) -> None:
        data = InferenceData()
        ctx = set_inference_context_data(data)
        self.assertIsNone(get_inference_context_data())
        with self.handler.inference(
            "downstream", request_model="gpt-4o", context=ctx
        ) as inv:
            self.assertIsInstance(inv, SuppressedInferenceInvocation)
            inv.input_tokens = 12
            inv.output_tokens = 24
        self.assertEqual(data.usage_input_tokens, 12)
        self.assertEqual(data.usage_output_tokens, 24)

    def test_finish_reasons_append_is_recorded(self) -> None:
        with self.handler.inference("openai", request_model="m") as inv:
            inv.finish_reasons = []
            inv.finish_reasons.append("stop")
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_RESPONSE_FINISH_REASONS),
            ("stop",),
        )

    def test_stop_sequences_append_is_recorded(self) -> None:
        with self.handler.inference("openai", request_model="m") as inv:
            inv.stop_sequences = []
            inv.stop_sequences.append("stop")
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_REQUEST_STOP_SEQUENCES), ("stop",)
        )

    @patch(
        "opentelemetry.util.genai.handler.get_content_capturing_mode",
        return_value=ContentCapturingMode.SPAN_AND_EVENT,
    )
    def test_tool_definitions_append_is_recorded(
        self, _mock_cap: object
    ) -> None:
        handler = TelemetryHandler(tracer_provider=self.tracer_provider)
        tool = FunctionToolDefinition(
            name="test_tool", description="a tool", parameters={}
        )
        with handler.inference("openai", request_model="m") as inv:
            inv.tool_definitions = []
            inv.tool_definitions.append(tool)
        self.assertEqual(inv.data.tool_definitions, [tool])
        (span,) = self.span_exporter.get_finished_spans()
        self.assertIn(GenAI.GEN_AI_TOOL_DEFINITIONS, span.attributes)

    def test_conversation_id_root_explicit_precedence_over_inner(self) -> None:
        with self.handler.inference(
            "proxy-provider", conversation_id="conv-root"
        ):
            with self.handler.inference(
                "downstream-provider", conversation_id="conv-inner"
            ) as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID), "conv-root"
        )

    def test_conversation_id_root_ambient_precedence_over_inner(self) -> None:
        token = attach(with_conversation_id("conv-ambient"))
        try:
            with self.handler.inference("proxy-provider"):
                with self.handler.inference(
                    "downstream-provider", conversation_id="conv-inner"
                ) as inner:
                    self.assertIsInstance(inner, SuppressedInferenceInvocation)
            (span,) = self.span_exporter.get_finished_spans()
            self.assertEqual(
                span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID),
                "conv-ambient",
            )
        finally:
            detach(token)

    def test_conversation_id_root_set_after_start_precedence_over_inner(
        self,
    ) -> None:
        with self.handler.inference("proxy-provider") as root:
            root.conversation_id = "conv-late-root"
            with self.handler.inference(
                "downstream-provider", conversation_id="conv-inner"
            ) as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID), "conv-late-root"
        )

    def test_conversation_id_inner_enriches_root_when_root_has_none(
        self,
    ) -> None:
        with self.handler.inference("proxy-provider"):
            with self.handler.inference(
                "downstream-provider", conversation_id="conv-inner"
            ) as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID), "conv-inner"
        )

    def test_conversation_id_inner_set_after_start_enriches_root(self) -> None:
        with self.handler.inference("proxy-provider"):
            with self.handler.inference("downstream-provider") as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
                inner.conversation_id = "conv-inner-late"
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_CONVERSATION_ID),
            "conv-inner-late",
        )

    def test_non_inference_data_in_context_does_not_suppress(self) -> None:
        ctx = set_value(CLIENT_INFERENCE_CONTEXT_KEY, "invalid-data")
        with self.handler.inference("openai", context=ctx) as inv:
            self.assertNotIsInstance(inv, SuppressedInferenceInvocation)
            self.assertIsInstance(inv, InferenceInvocation)


class TestEmbeddingContext(BaseContextTest):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            EMBEDDING_CONTEXT_KEY,
            "opentelemetry.genai.embedding.context",
        )

    def test_embedding_context_data_merge(self) -> None:
        data1 = EmbeddingData(
            request_model="model-1",
            embeddings_dimension_count=512,
            attributes={"k1": "v1"},
        )
        data2 = EmbeddingData(
            request_model="model-2",
            response_model="model-resp",
            usage_input_tokens=100,
            attributes={"k1": "v2", "k2": "v3"},
        )
        data1.merge(data2, overwrite=True)
        self.assertEqual(data1.request_model, "model-2")
        self.assertEqual(data1.embeddings_dimension_count, 512)
        self.assertEqual(data1.response_model, "model-resp")
        self.assertEqual(data1.usage_input_tokens, 100)
        self.assertEqual(data1.attributes, {"k1": "v2", "k2": "v3"})

        # overwrite=False preserves existing values
        data3 = EmbeddingData(
            request_model="root-model", embeddings_dimension_count=256
        )
        data4 = EmbeddingData(
            request_model="inner-model",
            response_model="inner-resp",
            usage_input_tokens=50,
        )
        data3.merge(data4, overwrite=False)
        self.assertEqual(data3.request_model, "root-model")
        self.assertEqual(data3.embeddings_dimension_count, 256)
        self.assertEqual(data3.response_model, "inner-resp")
        self.assertEqual(data3.usage_input_tokens, 50)

    def test_embedding_invocation_property_delegation_to_data(self) -> None:
        inv = self.handler.embedding(
            "test-provider",
            request_model="text-embedding-3-small",
            server_address="api.openai.com",
            server_port=443,
        )
        self.assertEqual(inv.provider, "test-provider")
        self.assertEqual(inv.request_model, "text-embedding-3-small")
        self.assertEqual(inv.server_address, "api.openai.com")
        self.assertEqual(inv.server_port, 443)

        inv.response_model_name = "text-embedding-3-small-v1"
        self.assertEqual(inv.data.response_model, "text-embedding-3-small-v1")
        self.assertEqual(inv.response_model_name, "text-embedding-3-small-v1")

        inv.dimension_count = 1536
        self.assertEqual(inv.data.embeddings_dimension_count, 1536)
        self.assertEqual(inv.dimension_count, 1536)

        inv.encoding_formats = ["float", "base64"]
        self.assertEqual(
            inv.data.request_encoding_formats, ["float", "base64"]
        )
        self.assertEqual(inv.encoding_formats, ["float", "base64"])

        inv.input_tokens = 42
        self.assertEqual(inv.data.usage_input_tokens, 42)
        self.assertEqual(inv.input_tokens, 42)
        inv.stop()

    def test_embedding_invocation_outer_sets_empty_object_on_context(
        self,
    ) -> None:
        self.assertIsNone(get_value(EMBEDDING_CONTEXT_KEY))
        with self.handler.embedding(
            "openai", request_model="text-embedding-3-small"
        ) as inv:
            self.assertIsInstance(inv, EmbeddingInvocation)
            self.assertNotIsInstance(inv, SuppressedEmbeddingInvocation)
            ctx_data = get_value(EMBEDDING_CONTEXT_KEY)
            self.assertIsInstance(ctx_data, EmbeddingData)
        self.assertIsNone(get_value(EMBEDDING_CONTEXT_KEY))

    def test_nested_embedding_deduplication_and_enrichment(self) -> None:
        with self.handler.embedding(
            "root-provider", request_model="root-model"
        ) as root:
            self.assertIsInstance(root, EmbeddingInvocation)
            self.assertNotIsInstance(root, SuppressedEmbeddingInvocation)

            with self.handler.embedding(
                "inner-provider", request_model="root-model"
            ) as inner:
                self.assertIsInstance(inner, SuppressedEmbeddingInvocation)
                inner.response_model_name = "actual-embed-model"
                inner.dimension_count = 1024
                inner.input_tokens = 50
                inner.attributes["custom.attr"] = "inner-val"

        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.name, "embeddings root-model")
        attrs = span.attributes
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_PROVIDER_NAME), "root-provider"
        )
        self.assertEqual(attrs.get(GenAI.GEN_AI_REQUEST_MODEL), "root-model")
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_RESPONSE_MODEL), "actual-embed-model"
        )
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT), 1024
        )
        self.assertEqual(attrs.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 50)
        self.assertEqual(attrs.get("custom.attr"), "inner-val")

    def test_root_precedence_over_nested_embedding(self) -> None:
        with self.handler.embedding(
            "root-provider",
            request_model="root-model",
            server_address="root.example.com",
            server_port=8080,
        ) as root:
            root.dimension_count = 512
            root.input_tokens = 10
            root.attributes["custom.attr"] = "root-val"

            with self.handler.embedding(
                "inner-provider",
                request_model="root-model",
                server_address="inner.example.com",
                server_port=443,
            ) as inner:
                inner.dimension_count = 1024
                inner.input_tokens = 100
                inner.response_model_name = "inner-resp"
                inner.attributes["custom.attr"] = "inner-val"

        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(
            attrs.get(server_attributes.SERVER_ADDRESS), "root.example.com"
        )
        self.assertEqual(attrs.get(server_attributes.SERVER_PORT), 8080)
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_EMBEDDINGS_DIMENSION_COUNT), 512
        )
        self.assertEqual(attrs.get(GenAI.GEN_AI_USAGE_INPUT_TOKENS), 10)
        self.assertEqual(attrs.get("custom.attr"), "root-val")
        self.assertEqual(attrs.get(GenAI.GEN_AI_RESPONSE_MODEL), "inner-resp")

    def test_nested_embedding_invocation_does_not_end_span_on_fail(
        self,
    ) -> None:
        with self.handler.embedding(
            "root-provider", request_model="root-model"
        ) as root:
            with self.assertRaises(ValueError):
                with self.handler.embedding(
                    "inner-provider", request_model="root-model"
                ) as inner:
                    self.assertIsInstance(inner, SuppressedEmbeddingInvocation)
                    raise ValueError("inner failure")
            self.assertTrue(root.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

    def test_nested_embedding_enrichment_applies_server_and_model_to_span(
        self,
    ) -> None:
        with self.handler.embedding("root-provider"):
            with self.handler.embedding(
                "inner-provider",
                request_model="inner-model",
                server_address="embed.inner.com",
                server_port=8443,
            ) as inner:
                self.assertIsInstance(inner, SuppressedEmbeddingInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(attrs.get(GenAI.GEN_AI_REQUEST_MODEL), "inner-model")
        self.assertEqual(
            attrs.get(server_attributes.SERVER_ADDRESS), "embed.inner.com"
        )
        self.assertEqual(attrs.get(server_attributes.SERVER_PORT), 8443)


class TestToolContext(BaseContextTest):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(TOOL_CONTEXT_KEY, "opentelemetry.genai.tool.context")

    def test_tool_context_data_merge(self) -> None:
        data1 = ToolData(tool_name="tool-1", tool_type="function")
        data2 = ToolData(
            tool_name="tool-2",
            tool_call_id="call-123",
            tool_description="desc",
        )
        data1.merge(data2, overwrite=True)
        self.assertEqual(data1.tool_name, "tool-2")
        self.assertEqual(data1.tool_type, "function")
        self.assertEqual(data1.tool_call_id, "call-123")
        self.assertEqual(data1.tool_description, "desc")

    def test_tool_invocation_property_delegation_to_data(self) -> None:
        inv = self.handler.tool(
            "calculator",
            tool_type="function",
            agent_name="MathAgent",
        )
        self.assertEqual(inv.name, "calculator")
        self.assertEqual(inv.tool_type, "function")
        self.assertEqual(inv.agent_name, "MathAgent")

        inv.tool_call_id = "call-abc"
        self.assertEqual(inv.data.tool_call_id, "call-abc")
        self.assertEqual(inv.tool_call_id, "call-abc")

        inv.tool_description = "A math tool"
        self.assertEqual(inv.data.tool_description, "A math tool")
        self.assertEqual(inv.tool_description, "A math tool")

        inv.arguments = {"expr": "1+1"}
        self.assertEqual(inv.data.tool_call_arguments, {"expr": "1+1"})
        self.assertEqual(inv.arguments, {"expr": "1+1"})

        inv.tool_result = 2
        self.assertEqual(inv.data.tool_call_result, 2)
        self.assertEqual(inv.tool_result, 2)
        inv.stop()

    def test_tool_invocation_outer_sets_empty_object_on_context(self) -> None:
        self.assertIsNone(get_value(TOOL_CONTEXT_KEY))
        with self.handler.tool("search") as inv:
            self.assertIsInstance(inv, ToolInvocation)
            self.assertNotIsInstance(inv, SuppressedToolInvocation)
            ctx_data = get_value(TOOL_CONTEXT_KEY)
            self.assertIsInstance(ctx_data, ToolData)
        self.assertIsNone(get_value(TOOL_CONTEXT_KEY))

    def test_nested_tool_deduplication_and_enrichment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "SPAN_ONLY"
            },
        ):
            with self.handler.tool("root-tool") as root:
                self.assertIsInstance(root, ToolInvocation)
                self.assertNotIsInstance(root, SuppressedToolInvocation)

                with self.handler.tool("root-tool") as inner:
                    self.assertIsInstance(inner, SuppressedToolInvocation)
                    inner.tool_call_id = "inner-call-id"
                    inner.tool_description = "inner tool desc"
                    inner.arguments = {"query": "test"}
                    inner.tool_result = "result"

            (span,) = self.span_exporter.get_finished_spans()
            attrs = span.attributes
            self.assertEqual(attrs.get(GenAI.GEN_AI_TOOL_NAME), "root-tool")
            self.assertEqual(
                attrs.get(GenAI.GEN_AI_TOOL_CALL_ID), "inner-call-id"
            )
            self.assertEqual(
                attrs.get(GenAI.GEN_AI_TOOL_DESCRIPTION), "inner tool desc"
            )

    def test_enrich_from_context_does_not_override_content(self) -> None:
        with self.handler.tool("root-tool") as outer:
            outer.data.tool_call_arguments = {"root_arg": 1}
            outer.data.tool_call_result = "root_res"
            outer.data.tool_description = "root_desc"

            inner_data = ToolData(
                tool_call_id="call-123",
                tool_description="inner_desc",
                tool_call_arguments={"inner_arg": 2},
                tool_call_result="inner_res",
            )
            outer.enrich_from_context(inner_data)

            # Non-content fields are enriched
            self.assertEqual(outer.data.tool_call_id, "call-123")
            # Outer non-content fields take precedence
            self.assertEqual(outer.data.tool_description, "root_desc")
            # Content fields are NOT overridden by inner
            self.assertEqual(outer.data.tool_call_arguments, {"root_arg": 1})
            self.assertEqual(outer.data.tool_call_result, "root_res")

    def test_nested_tool_invocation_does_not_end_span_on_fail(self) -> None:
        with self.handler.tool("root-tool") as root:
            with self.assertRaises(ValueError):
                with self.handler.tool("root-tool") as inner:
                    self.assertIsInstance(inner, SuppressedToolInvocation)
                    raise ValueError("inner failure")
            self.assertTrue(root.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

    def test_nested_tool_enrichment_applies_tool_type_to_span(self) -> None:
        with self.handler.tool("root-tool"):
            with self.handler.tool("root-tool", tool_type="function") as inner:
                self.assertIsInstance(inner, SuppressedToolInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(
            span.attributes.get(GenAI.GEN_AI_TOOL_TYPE), "function"
        )

    def test_nested_tool_different_name_not_suppressed_and_context_restored(
        self,
    ) -> None:
        with self.handler.tool("outer-tool") as outer:
            self.assertIsInstance(outer, ToolInvocation)
            self.assertNotIsInstance(outer, SuppressedToolInvocation)
            outer_data = get_value(TOOL_CONTEXT_KEY)
            self.assertIsInstance(outer_data, ToolData)
            self.assertEqual(outer_data.tool_name, "outer-tool")

            with self.handler.tool("inner-tool") as inner:
                self.assertIsInstance(inner, ToolInvocation)
                self.assertNotIsInstance(inner, SuppressedToolInvocation)
                inner_data = get_value(TOOL_CONTEXT_KEY)
                self.assertIsInstance(inner_data, ToolData)
                self.assertEqual(inner_data.tool_name, "inner-tool")

            # After inner tool exits, outer tool context is restored
            restored_data = get_value(TOOL_CONTEXT_KEY)
            self.assertIs(restored_data, outer_data)
            self.assertEqual(restored_data.tool_name, "outer-tool")

        # After outer tool exits, context is cleared
        self.assertIsNone(get_value(TOOL_CONTEXT_KEY))
        spans = self.span_exporter.get_finished_spans()
        self.assertEqual(len(spans), 2)
        span_names = [s.name for s in spans]
        self.assertEqual(
            span_names, ["execute_tool inner-tool", "execute_tool outer-tool"]
        )


class TestRetrievalContext(BaseContextTest):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            RETRIEVAL_CONTEXT_KEY,
            "opentelemetry.genai.retrieval.context",
        )

    def test_retrieval_context_data_merge(self) -> None:
        data1 = RetrievalData(data_source_id="ds-1", retrieval_top_k=5)
        data2 = RetrievalData(
            data_source_id="ds-2",
            retrieval_top_k=10,
            retrieval_query_text="search query",
        )
        data1.merge(data2, overwrite=True)
        self.assertEqual(data1.data_source_id, "ds-2")
        self.assertEqual(data1.retrieval_top_k, 10)
        self.assertEqual(data1.retrieval_query_text, "search query")

    def test_retrieval_invocation_property_delegation_to_data(self) -> None:
        inv = self.handler.retrieval(
            data_source_id="my-ds",
            provider="chroma",
            request_model="all-MiniLM-L6-v2",
            server_address="localhost",
            server_port=8000,
        )
        self.assertEqual(inv.data_source_id, "my-ds")
        self.assertEqual(inv.provider, "chroma")
        self.assertEqual(inv.request_model, "all-MiniLM-L6-v2")
        self.assertEqual(inv.server_address, "localhost")
        self.assertEqual(inv.server_port, 8000)

        inv.top_k = 15
        self.assertEqual(inv.data.retrieval_top_k, 15)
        self.assertEqual(inv.top_k, 15)

        inv.query_text = "What is OpenTelemetry?"
        self.assertEqual(
            inv.data.retrieval_query_text, "What is OpenTelemetry?"
        )
        self.assertEqual(inv.query_text, "What is OpenTelemetry?")

        doc = RetrievalDocument(id="doc-1", score=0.8)
        inv.documents = [doc]
        self.assertEqual(inv.data.retrieval_documents, [doc])
        self.assertEqual(inv.documents, [doc])
        inv.stop()

    def test_retrieval_invocation_outer_sets_empty_object_on_context(
        self,
    ) -> None:
        self.assertIsNone(get_value(RETRIEVAL_CONTEXT_KEY))
        with self.handler.retrieval(data_source_id="ds-1") as inv:
            self.assertIsInstance(inv, RetrievalInvocation)
            self.assertNotIsInstance(inv, SuppressedRetrievalInvocation)
            ctx_data = get_value(RETRIEVAL_CONTEXT_KEY)
            self.assertIsInstance(ctx_data, RetrievalData)
        self.assertIsNone(get_value(RETRIEVAL_CONTEXT_KEY))

    def test_nested_retrieval_deduplication_and_enrichment(self) -> None:
        with self.handler.retrieval(data_source_id="root-ds") as root:
            self.assertIsInstance(root, RetrievalInvocation)
            self.assertNotIsInstance(root, SuppressedRetrievalInvocation)

            with self.handler.retrieval(data_source_id="root-ds") as inner:
                self.assertIsInstance(inner, SuppressedRetrievalInvocation)
                inner.top_k = 20
                inner.attributes["custom.retrieval"] = "val"

        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(attrs.get(GenAI.GEN_AI_DATA_SOURCE_ID), "root-ds")
        self.assertEqual(attrs.get("gen_ai.retrieval.top_k"), 20)
        self.assertEqual(attrs.get("custom.retrieval"), "val")

    def test_enrich_from_context_does_not_override_content(self) -> None:
        with self.handler.retrieval(data_source_id="root-ds") as outer:
            outer.data.retrieval_query_text = "root query"
            outer.data.retrieval_documents = [
                RetrievalDocument(id="root-doc", score=0.9)
            ]
            outer.data.retrieval_top_k = 5

            inner_data = RetrievalData(
                retrieval_top_k=10,
                retrieval_query_text="inner query",
                retrieval_documents=[
                    RetrievalDocument(id="inner-doc", score=0.1)
                ],
                data_source_id="inner-ds",
            )
            outer.enrich_from_context(inner_data)

            # Non-content fields: outer top_k preserved
            self.assertEqual(outer.data.retrieval_top_k, 5)
            # Inner non-content fields not on outer are enriched
            self.assertEqual(outer.data.data_source_id, "root-ds")
            # Content fields are NOT overridden by inner
            self.assertEqual(outer.data.retrieval_query_text, "root query")
            self.assertEqual(
                outer.data.retrieval_documents,
                [RetrievalDocument(id="root-doc", score=0.9)],
            )

    def test_nested_retrieval_invocation_does_not_end_span_on_fail(
        self,
    ) -> None:
        with self.handler.retrieval(data_source_id="root-ds") as root:
            with self.assertRaises(ValueError):
                with self.handler.retrieval(data_source_id="root-ds") as inner:
                    self.assertIsInstance(inner, SuppressedRetrievalInvocation)
                    raise ValueError("inner failure")
            self.assertTrue(root.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

    def test_nested_retrieval_enrichment_applies_non_content_fields_to_span(
        self,
    ) -> None:
        with self.handler.retrieval():
            with self.handler.retrieval(
                data_source_id="inner-ds",
                provider="inner-provider",
                request_model="inner-model",
                server_address="search.inner.com",
                server_port=9200,
            ) as inner:
                self.assertIsInstance(inner, SuppressedRetrievalInvocation)
                inner.top_k = 5
        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(attrs.get(GenAI.GEN_AI_DATA_SOURCE_ID), "inner-ds")
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_PROVIDER_NAME), "inner-provider"
        )
        self.assertEqual(attrs.get(GenAI.GEN_AI_REQUEST_MODEL), "inner-model")
        self.assertEqual(
            attrs.get(server_attributes.SERVER_ADDRESS), "search.inner.com"
        )
        self.assertEqual(attrs.get(server_attributes.SERVER_PORT), 9200)
        self.assertEqual(attrs.get("gen_ai.retrieval.top_k"), 5)


class TestFetchResponseContext(BaseContextTest):
    def setUp(self) -> None:
        super().setUp()
        self.span_exporter = InMemorySpanExporter()
        self.tracer_provider.add_span_processor(
            SimpleSpanProcessor(self.span_exporter)
        )
        self.handler = TelemetryHandler(
            tracer_provider=self.tracer_provider,
            meter_provider=self.meter_provider,
        )

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            FETCH_RESPONSE_CONTEXT_KEY,
            "opentelemetry.genai.fetch_response.context",
        )

    def test_fetch_response_context_data_merge(self) -> None:
        data1 = FetchResponseData(
            response_id="resp-1", response_status="pending"
        )
        data2 = FetchResponseData(
            response_id="resp-2",
            response_status="completed",
            response_model="gpt-4o",
        )
        data1.merge(data2, overwrite=True)
        self.assertEqual(data1.response_id, "resp-2")
        self.assertEqual(data1.response_status, "completed")
        self.assertEqual(data1.response_model, "gpt-4o")

    def test_fetch_response_invocation_property_delegation_to_data(
        self,
    ) -> None:
        inv = self.handler.fetch_response(
            "openai",
            response_id="resp-original",
            server_address="api.openai.com",
            server_port=443,
        )
        self.assertEqual(inv.provider, "openai")
        self.assertEqual(inv.response_id, "resp-original")
        self.assertEqual(inv.server_address, "api.openai.com")
        self.assertEqual(inv.server_port, 443)

        inv.response_model_name = "gpt-4o-mini"
        self.assertEqual(inv.data.response_model, "gpt-4o-mini")
        self.assertEqual(inv.response_model_name, "gpt-4o-mini")

        inv.response_status = "completed"
        self.assertEqual(inv.data.response_status, "completed")
        self.assertEqual(inv.response_status, "completed")

        inv.finish_reasons = ["stop"]
        self.assertEqual(inv.data.response_finish_reasons, ["stop"])
        self.assertEqual(inv.finish_reasons, ["stop"])

        inv.stream_cursor = "cursor-123"
        self.assertEqual(inv.data.request_stream_cursor, "cursor-123")
        self.assertEqual(inv.stream_cursor, "cursor-123")
        inv.stop()

    def test_fetch_response_invocation_outer_sets_empty_object_on_context(
        self,
    ) -> None:
        self.assertIsNone(get_value(FETCH_RESPONSE_CONTEXT_KEY))
        with self.handler.fetch_response(
            "openai", response_id="resp-root"
        ) as inv:
            self.assertIsInstance(inv, FetchResponseInvocation)
            self.assertNotIsInstance(inv, SuppressedFetchResponseInvocation)
            ctx_data = get_value(FETCH_RESPONSE_CONTEXT_KEY)
            self.assertIsInstance(ctx_data, FetchResponseData)
        self.assertIsNone(get_value(FETCH_RESPONSE_CONTEXT_KEY))

    def test_nested_fetch_response_deduplication_and_enrichment(self) -> None:
        with self.handler.fetch_response(
            "openai", response_id="resp-root"
        ) as root:
            self.assertIsInstance(root, FetchResponseInvocation)
            self.assertNotIsInstance(root, SuppressedFetchResponseInvocation)

            with self.handler.fetch_response(
                "openai", response_id="resp-root"
            ) as inner:
                self.assertIsInstance(inner, SuppressedFetchResponseInvocation)
                inner.response_model_name = "gpt-4o-2024-08-06"
                inner.response_status = "completed"
                inner.finish_reasons = ["stop"]
                inner.attributes["custom.fetch"] = "val"

        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(attrs.get(GenAI.GEN_AI_RESPONSE_ID), "resp-root")
        self.assertEqual(
            attrs.get(GenAI.GEN_AI_RESPONSE_MODEL), "gpt-4o-2024-08-06"
        )
        self.assertEqual(attrs.get("gen_ai.response.status"), "completed")
        self.assertEqual(
            list(attrs.get(GenAI.GEN_AI_RESPONSE_FINISH_REASONS) or []),
            ["stop"],
        )
        self.assertEqual(attrs.get("custom.fetch"), "val")

    def test_enrich_from_context_does_not_override_content(self) -> None:
        with self.handler.fetch_response(
            "openai", response_id="resp-root"
        ) as outer:
            outer.data.output_messages = [
                OutputMessage(role="assistant", parts=[TextPart("root out")])
            ]
            outer.data.response_model = "root-model"

            inner_data = FetchResponseData(
                response_model="inner-model",
                response_status="completed",
                output_messages=[
                    OutputMessage(
                        role="assistant", parts=[TextPart("inner out")]
                    )
                ],
            )
            outer.enrich_from_context(inner_data)

            # Non-content fields: outer preserved, inner enriched
            self.assertEqual(outer.data.response_model, "root-model")
            self.assertEqual(outer.data.response_status, "completed")
            # Content fields are NOT overridden by inner
            self.assertEqual(
                outer.data.output_messages,
                [
                    OutputMessage(
                        role="assistant", parts=[TextPart("root out")]
                    )
                ],
            )

    def test_nested_fetch_response_invocation_does_not_end_span_on_fail(
        self,
    ) -> None:
        with self.handler.fetch_response(
            "openai", response_id="resp-root"
        ) as root:
            with self.assertRaises(ValueError):
                with self.handler.fetch_response(
                    "openai", response_id="resp-root"
                ) as inner:
                    self.assertIsInstance(
                        inner, SuppressedFetchResponseInvocation
                    )
                    raise ValueError("fetch failure")
            self.assertTrue(root.span.is_recording())
            self.assertEqual(len(self.span_exporter.get_finished_spans()), 0)

        (span,) = self.span_exporter.get_finished_spans()
        self.assertEqual(span.status.status_code, StatusCode.UNSET)

    def test_nested_fetch_response_enrichment_applies_stream_and_server_to_span(
        self,
    ) -> None:
        with self.handler.fetch_response(
            "root-provider", response_id="resp-1"
        ):
            with self.handler.fetch_response(
                "inner-provider",
                response_id="resp-1",
                server_address="fetch.inner.com",
                server_port=443,
            ) as inner:
                inner.stream_cursor = "3"
                self.assertIsInstance(inner, SuppressedFetchResponseInvocation)
        (span,) = self.span_exporter.get_finished_spans()
        attrs = span.attributes
        self.assertEqual(
            attrs.get(server_attributes.SERVER_ADDRESS), "fetch.inner.com"
        )
        self.assertEqual(attrs.get(server_attributes.SERVER_PORT), 443)
        self.assertEqual(attrs.get("gen_ai.request.stream_cursor"), "3")

    def test_suppressed_fetch_response_does_not_emit_streaming_metrics(
        self,
    ) -> None:
        with self.handler.fetch_response(
            "root-provider", response_id="resp-1"
        ):
            with self.handler.fetch_response(
                "inner-provider", response_id="resp-1"
            ) as inner:
                self.assertIsInstance(inner, SuppressedFetchResponseInvocation)
                inner.record_stream_chunk()
                inner.record_stream_chunk()
                self.assertTrue(inner.data.request_stream)
        metrics = self._harvest_metrics()
        self.assertNotIn(
            "gen_ai.client.operation.time_to_first_chunk", metrics
        )
        self.assertNotIn(
            "gen_ai.client.operation.time_per_output_chunk", metrics
        )
