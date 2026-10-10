# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from opentelemetry import context as context_api

from .base import TestCase


class TestInteractionsAsync(TestCase):
    def run_interaction(self, *args: Any, **kwargs: Any) -> Any:
        async def _run() -> Any:
            # asyncio.run() discards the task's context, so a leak is only
            # visible from inside the task.
            before = context_api.get_current()
            try:
                return await self.client.aio.interactions.create(
                    *args, **kwargs
                )
            finally:
                self.assertIs(context_api.get_current(), before)

        return asyncio.run(_run())

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
