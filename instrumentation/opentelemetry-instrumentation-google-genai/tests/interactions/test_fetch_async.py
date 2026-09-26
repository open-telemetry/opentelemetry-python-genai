# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

from .fetch_base import TestCase


class TestFetchResponseAsync(TestCase):
    def run_fetch(self, *args: Any, **kwargs: Any) -> Any:
        return asyncio.run(self.client.aio.interactions.get(*args, **kwargs))

    def run_streaming_fetch(self, *args: Any, **kwargs: Any) -> list[Any]:
        async def _run() -> list[Any]:
            stream = await self.client.aio.interactions.get(*args, **kwargs)
            return [event async for event in stream]

        return asyncio.run(_run())

    def run_streaming_fetch_with_caller_error(
        self, *args: Any, **kwargs: Any
    ) -> None:
        async def _run() -> None:
            stream = await self.client.aio.interactions.get(*args, **kwargs)
            async with stream:
                raise RuntimeError("caller blew up")

        asyncio.run(_run())

    def drain_stream(self, *args: Any, **kwargs: Any) -> Any:
        async def _run() -> Any:
            stream = await self.client.aio.interactions.get(*args, **kwargs)
            [event async for event in stream]
            return stream

        return asyncio.run(_run())

    def run_streaming_fetch_closed_early(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        async def _run() -> Any:
            stream = await self.client.aio.interactions.get(*args, **kwargs)
            first = await anext(aiter(stream))
            await stream.aclose()
            return first

        return asyncio.run(_run())
