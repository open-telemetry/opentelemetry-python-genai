# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from unittest.mock import patch

from opentelemetry.context import Context, attach, detach
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
from opentelemetry.util.genai._context import (
    INFERENCE_CONTEXT_KEY,
    InferenceNonContentCaptureData,
    get_inference_context_data,
    set_inference_context_data,
)
from opentelemetry.util.genai._inference_invocation import (
    SuppressedInferenceInvocation,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    InferenceContentData,
    InferenceInvocation,
)


class TestInferenceContext(TestBase):
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

    def _harvest_metrics(self) -> dict[str, list[object]]:
        metrics = self.get_sorted_metrics()
        metrics_by_name: dict[str, list[object]] = {}
        for metric in metrics or []:
            points = getattr(metric.data, "data_points", None) or []
            metrics_by_name.setdefault(metric.name, []).extend(points)
        return metrics_by_name

    def test_context_key_constant_value(self) -> None:
        self.assertEqual(
            INFERENCE_CONTEXT_KEY,
            "opentelemetry.genai.inference_context",
        )

    def test_get_inference_context_data_none_by_default(self) -> None:
        self.assertIsNone(get_inference_context_data())

    def test_set_and_get_inference_context_data(self) -> None:
        data = InferenceNonContentCaptureData(request_model="gpt-4o")
        ctx = set_inference_context_data(data)
        self.assertIs(get_inference_context_data(ctx), data)
        self.assertIsNone(get_inference_context_data())

        token = attach(ctx)
        try:
            self.assertIs(get_inference_context_data(), data)
        finally:
            detach(token)
        self.assertIsNone(get_inference_context_data())

    def test_in_place_mutation_of_inference_context_data(self) -> None:
        data = InferenceNonContentCaptureData(input_tokens=1)
        ctx = set_inference_context_data(data)
        token = attach(ctx)
        try:
            current = get_inference_context_data()
            self.assertIsNotNone(current)
            assert current is not None
            current.input_tokens = 10
            current.output_tokens = 20

            after = get_inference_context_data()
            assert after is not None
            self.assertEqual(after.input_tokens, 10)
            self.assertEqual(after.output_tokens, 20)
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
            self.assertEqual(data, InferenceNonContentCaptureData())

            # Setting fields on outer does not mutate context object
            invocation.data.input_tokens = 42
            invocation.data.output_tokens = 84
            invocation.data.temperature = 0.7
            invocation.data.response_model = "gpt-4o-mini-2024-07-18"
            self.assertEqual(data, InferenceNonContentCaptureData())

        self.assertIsNone(get_inference_context_data())

    def test_inference_invocation_automatic_publish_on_finish(self) -> None:
        with self.handler.inference(
            "openai", request_model="gpt-4o-mini"
        ) as invocation:
            self.assertNotIsInstance(invocation, SuppressedInferenceInvocation)
            invocation.data.input_tokens = 10
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
                    nested_inv.data.input_tokens = 15
                    nested_inv.data.output_tokens = 25
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
                self.assertEqual(data.input_tokens, 15)
                self.assertEqual(data.output_tokens, 25)
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

    def test_llm_invocation_suppressed(self) -> None:
        from opentelemetry.util.genai._inference_invocation import (
            LLMInvocation,
        )

        with self.handler.inference("upstream"):
            nested_inv = LLMInvocation(request_model="nested")
            self.handler.start_llm(nested_inv)
            self.assertTrue(nested_inv.span.is_recording())
            assert nested_inv._inference_invocation is not None
            self.assertIsInstance(
                nested_inv._inference_invocation,
                SuppressedInferenceInvocation,
            )
            self.assertFalse(
                nested_inv._inference_invocation.should_capture_content
            )
            nested_inv.attributes["custom.llm"] = "val"
            self.handler.stop_llm(nested_inv)
            data = get_inference_context_data()
            self.assertIsNotNone(data)
            assert data is not None
            self.assertEqual(data.request_model, "nested")
            self.assertEqual(data.attributes.get("custom.llm"), "val")

    def test_metric_enrichment_precedence_and_error(self) -> None:
        with self.assertRaises(ValueError):
            with self.handler.inference(
                "proxy",
                request_model="gpt-4o",
                server_address="proxy.internal",
                server_port=8080,
            ) as root_inv:
                root_inv.data.input_tokens = 10
                with self.handler.inference(
                    "openai",
                    request_model="gpt-4o",
                    server_address="api.openai.com",
                    server_port=443,
                ) as inner_inv:
                    inner_inv.data.input_tokens = 99
                    inner_inv.data.output_tokens = 50
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
            root.data.temperature = 0.2
            root.data.input_tokens = 10
            root.attributes["custom.shared"] = "root-value"

            with self.handler.inference(
                "downstream-provider",
                request_model="downstream-model",
                server_address="api.example.com",
                server_port=443,
            ) as inner:
                self.assertIsInstance(inner, SuppressedInferenceInvocation)
                # Overwrite shared fields downstream
                inner.data.temperature = 0.9
                inner.data.input_tokens = 100
                inner.data.output_tokens = 50
                inner.data.response_id = "resp-123"
                inner.attributes["custom.shared"] = "downstream-value"
                inner.attributes["custom.downstream_only"] = "downstream-only"

            # While still in root context, context reflects downstream writes
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.temperature, 0.9)
            self.assertEqual(data.input_tokens, 100)
            self.assertEqual(data.output_tokens, 50)
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
                InferenceNonContentCaptureData(),
            )

            with self.handler.inference(
                "inner1-provider",
                request_model="model-1",
            ) as inner1:
                self.assertIsInstance(inner1, SuppressedInferenceInvocation)
                inner1.data.input_tokens = 10
                inner1.data.output_tokens = 20
                inner1.data.response_model = "resp-model-1"

            # After inner1 finishes, context has inner1's attributes
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.request_model, "model-1")
            self.assertEqual(data.input_tokens, 10)
            self.assertEqual(data.response_model, "resp-model-1")

            with self.handler.inference(
                "inner2-provider",
                request_model="model-2",
            ) as inner2:
                self.assertIsInstance(inner2, SuppressedInferenceInvocation)
                inner2.data.input_tokens = 30
                inner2.data.output_tokens = 40
                inner2.data.response_model = "resp-model-2"

            # After inner2 finishes, inner2 overwrites inner1
            data = get_inference_context_data()
            assert data is not None
            self.assertEqual(data.request_model, "model-2")
            self.assertEqual(data.input_tokens, 30)
            self.assertEqual(data.output_tokens, 40)
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
            assert data is not None
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
            inner_inv.data.input_tokens = 15
            inner_inv.data.output_tokens = 25
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
            self.assertEqual(data.input_tokens, 15)
            self.assertEqual(data.output_tokens, 25)
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
                inner.data.temperature = 0.7
                inner.data.top_p = 0.95
                inner.data.top_k = 40
                inner.data.frequency_penalty = 0.5
                inner.data.presence_penalty = 0.2
                inner.data.max_tokens = 2048
                inner.data.stop_sequences = ["\n", "STOP"]
                inner.data.seed = 42
                inner.data.request_choice_count = 1
                inner.data.output_type = "json"
                inner.data.response_id = "resp-xyz"
                inner.data.finish_reasons = ["stop"]
                inner.data.input_tokens = 120
                inner.data.output_tokens = 60
                inner.data.thinking_tokens = 30
                inner.data.cache_read_input_tokens = 10
                inner.data.cache_write_input_tokens = 5
                inner.data.reasoning_level = "high"
                inner.data.prompt_name = "test-prompt"
                inner.data.prompt_version = "v1"
                inner.attributes["custom.foo"] = "bar"

            data = get_inference_context_data()
            self.assertIsNotNone(data)
            assert data is not None
            # Standard non-content attributes are on typed fields
            self.assertEqual(data.temperature, 0.7)
            self.assertEqual(data.top_p, 0.95)
            self.assertEqual(data.top_k, 40)
            self.assertEqual(data.frequency_penalty, 0.5)
            self.assertEqual(data.presence_penalty, 0.2)
            self.assertEqual(data.max_tokens, 2048)
            self.assertEqual(data.stop_sequences, ["\n", "STOP"])
            self.assertEqual(data.seed, 42)
            self.assertEqual(data.request_choice_count, 1)
            self.assertEqual(data.output_type, "json")
            self.assertEqual(data.response_id, "resp-xyz")
            self.assertEqual(data.finish_reasons, ["stop"])
            self.assertEqual(data.input_tokens, 120)
            self.assertEqual(data.output_tokens, 60)
            self.assertEqual(data.thinking_tokens, 30)
            self.assertEqual(data.cache_read_input_tokens, 10)
            self.assertEqual(data.cache_write_input_tokens, 5)
            self.assertEqual(data.reasoning_level, "high")
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
            # Underlying dataclass instances exist
            self.assertIsInstance(inv.data, InferenceNonContentCaptureData)
            self.assertIsInstance(inv.content, InferenceContentData)

            # Dictionary references are shared
            self.assertIs(inv.attributes, inv.data.attributes)
            self.assertIs(inv.metric_attributes, inv.data.metric_attributes)

            # Context fields are on inv.data
            inv.data.input_tokens = 100
            self.assertEqual(inv.data.input_tokens, 100)
            inv.data.output_tokens = 50
            self.assertEqual(inv.data.output_tokens, 50)
            inv.data.response_model = "gpt-4o-2024-08-06"
            self.assertEqual(inv.data.response_model, "gpt-4o-2024-08-06")

            # Content access on inv.content
            from opentelemetry.util.genai.types import InputMessage, TextPart

            inv.content.input_messages.append(
                InputMessage(role="user", parts=[TextPart(content="hello")])
            )
            self.assertEqual(len(inv.content.input_messages), 1)

            # Content fields are not present on inv.data
            self.assertFalse(hasattr(inv.data, "input_messages"))
            self.assertFalse(hasattr(inv.data, "output_messages"))
            self.assertFalse(hasattr(inv.data, "system_instruction"))
            self.assertFalse(hasattr(inv.data, "tool_definitions"))

    def test_inference_context_data_merge(self) -> None:
        base = InferenceNonContentCaptureData(
            provider="openai",
            request_model="gpt-4o",
            temperature=0.5,
            input_tokens=10,
            attributes={"a": 1, "shared": "base"},
            metric_attributes={"m1": "v1"},
        )
        incoming = InferenceNonContentCaptureData(
            request_model="gpt-4o-mini",
            temperature=0.9,
            output_tokens=20,
            attributes={"b": 2, "shared": "incoming"},
            metric_attributes={"m2": "v2"},
        )

        # Merge with overwrite=True (inner publishing to context)
        dest = InferenceNonContentCaptureData()
        dest.merge(base, overwrite=True)
        dest.merge(incoming, overwrite=True)
        self.assertEqual(dest.provider, "openai")
        self.assertEqual(dest.request_model, "gpt-4o-mini")  # overwritten
        self.assertEqual(dest.temperature, 0.9)  # overwritten
        self.assertEqual(dest.input_tokens, 10)
        self.assertEqual(dest.output_tokens, 20)
        self.assertEqual(dest.attributes["shared"], "incoming")  # overwritten
        self.assertEqual(dest.attributes["a"], 1)
        self.assertEqual(dest.attributes["b"], 2)
        self.assertEqual(dest.metric_attributes["m1"], "v1")
        self.assertEqual(dest.metric_attributes["m2"], "v2")

        # Merge with overwrite=False (outer enriching from context)
        outer = InferenceNonContentCaptureData(
            provider="openai",
            request_model="gpt-4o",
            temperature=0.2,  # outer explicitly set temperature
            attributes={"custom": "outer", "shared": "outer"},
        )
        outer.merge(dest, overwrite=False)
        self.assertEqual(outer.request_model, "gpt-4o")  # preserved outer
        self.assertEqual(outer.temperature, 0.2)  # preserved outer
        self.assertEqual(outer.input_tokens, 10)  # enriched from inner
        self.assertEqual(outer.output_tokens, 20)  # enriched from inner
        self.assertEqual(
            outer.attributes["shared"], "outer"
        )  # preserved outer
        self.assertEqual(outer.attributes["a"], 1)  # enriched from inner
