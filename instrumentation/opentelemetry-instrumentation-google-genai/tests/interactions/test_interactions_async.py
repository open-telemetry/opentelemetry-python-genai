# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry.semconv.attributes import server_attributes
from opentelemetry.util.genai.handler import TelemetryHandler

from .base import TestCase


class TestInteractionsAsync(TestCase):
    def run_interaction(self, *args: Any, **kwargs: Any) -> Any:
        return asyncio.run(
            self.client.aio.interactions.create(*args, **kwargs)
        )

    def run_streaming_interaction(
        self, *args: Any, **kwargs: Any
    ) -> list[Any]:
        async def _run() -> list[Any]:
            stream = await self.client.aio.interactions.create(*args, **kwargs)
            events = []
            async for event in stream:
                events.append(event)
            return events

        return asyncio.run(_run())

    def test_async_interactions_create_suppressed(self) -> None:
        async def _run() -> None:
            self.configure_valid_interaction()

            upstream_handler = TelemetryHandler()
            with upstream_handler.inference(
                "upstream-agent", request_model="gemini-2.0-flash"
            ):
                response = await self.client.aio.interactions.create(
                    model="gemini-2.0-flash",
                    input="Hello async interaction",
                )
                self.assertIsNotNone(response)

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
