# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os

from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.util.genai.handler import TelemetryHandler

from .base import TestCase


class TestInferenceSuppression(TestCase):
    def setUp(self):
        super().setUp()
        os.environ["OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT"] = "true"

    def test_sync_generate_content_suppressed(self):
        self.configure_valid_response(text="Hello world from Gemini")

        upstream_handler = TelemetryHandler()
        with upstream_handler.inference(
            "upstream-agent", request_model="gemini-2.0-flash"
        ):
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )
            self.assertEqual(response.text, "Hello world from Gemini")

        spans = self.otel.get_finished_spans()
        logs = self.otel.get_finished_logs()

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(logs), 1)

        span = spans[0]
        self.assertEqual(
            span.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            span.attributes["google_genai.inference_suppressed"], True
        )

        log = logs[0]
        self.assertEqual(
            log.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            log.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            log.attributes["google_genai.inference_suppressed"], True
        )

    def test_sync_streaming_generate_content_suppressed(self):
        self.configure_valid_response(text="Chunk 1")
        self.configure_valid_response(text="Chunk 2")

        upstream_handler = TelemetryHandler()
        with upstream_handler.inference(
            "upstream-agent", request_model="gemini-2.0-flash"
        ):
            stream = self.client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents="Hello stream",
            )
            chunks = list(stream)
            self.assertEqual(len(chunks), 2)

        spans = self.otel.get_finished_spans()
        logs = self.otel.get_finished_logs()

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(logs), 1)

        span = spans[0]
        self.assertEqual(
            span.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            span.attributes["google_genai.inference_suppressed"], True
        )

        log = logs[0]
        self.assertEqual(
            log.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            log.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            log.attributes["google_genai.inference_suppressed"], True
        )

    def test_async_generate_content_suppressed(self):
        async def _run():
            self.configure_valid_response(text="Async hello")
            upstream_handler = TelemetryHandler()
            with upstream_handler.inference(
                "upstream-agent", request_model="gemini-2.0-flash"
            ):
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.0-flash",
                    contents="Hello async",
                )
                self.assertEqual(response.text, "Async hello")

        asyncio.run(_run())

        spans = self.otel.get_finished_spans()
        logs = self.otel.get_finished_logs()

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(logs), 1)

        span = spans[0]
        self.assertEqual(
            span.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            span.attributes["google_genai.inference_suppressed"], True
        )

        log = logs[0]
        self.assertEqual(
            log.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            log.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            log.attributes["google_genai.inference_suppressed"], True
        )

    def test_async_streaming_generate_content_suppressed(self):
        async def _run():
            self.configure_valid_response(text="Async chunk 1")
            upstream_handler = TelemetryHandler()
            with upstream_handler.inference(
                "upstream-agent", request_model="gemini-2.0-flash"
            ):
                stream = await self.client.aio.models.generate_content_stream(
                    model="gemini-2.0-flash",
                    contents="Hello async stream",
                )
                chunks = []
                async for c in stream:
                    chunks.append(c)
                self.assertEqual(len(chunks), 1)

        asyncio.run(_run())

        spans = self.otel.get_finished_spans()
        logs = self.otel.get_finished_logs()

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(logs), 1)

        span = spans[0]
        self.assertEqual(
            span.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            span.attributes["google_genai.inference_suppressed"], True
        )

        log = logs[0]
        self.assertEqual(
            log.attributes[server_attributes.SERVER_ADDRESS],
            "generativelanguage.googleapis.com",
        )
        self.assertEqual(
            log.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            log.attributes["google_genai.inference_suppressed"], True
        )

    def test_vertex_ai_enrichment_suppressed(self):
        self.set_use_vertex(True)
        self.configure_valid_response(text="Hello vertex")

        upstream_handler = TelemetryHandler()
        with upstream_handler.inference(
            "upstream-agent", request_model="gemini-2.0-flash"
        ):
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Hello",
            )
            self.assertEqual(response.text, "Hello vertex")

        spans = self.otel.get_finished_spans()
        logs = self.otel.get_finished_logs()

        self.assertEqual(len(spans), 1)
        self.assertEqual(len(logs), 1)

        span = spans[0]
        self.assertEqual(
            span.attributes[server_attributes.SERVER_ADDRESS],
            "test-location-aiplatform.googleapis.com",
        )
        self.assertEqual(
            span.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            span.attributes["google_genai.inference_suppressed"], True
        )

        log = logs[0]
        self.assertEqual(
            log.attributes[server_attributes.SERVER_ADDRESS],
            "test-location-aiplatform.googleapis.com",
        )
        self.assertEqual(
            log.attributes["gen_ai.provider.name"], "upstream-agent"
        )
        self.assertEqual(
            log.attributes["google_genai.inference_suppressed"], True
        )
