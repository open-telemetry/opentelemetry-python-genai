# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from .fetch_base import TestCase


class TestFetchResponseSync(TestCase):
    def run_fetch(self, *args: Any, **kwargs: Any) -> Any:
        return self.client.interactions.get(*args, **kwargs)

    def run_streaming_fetch(self, *args: Any, **kwargs: Any) -> list[Any]:
        return list(self.client.interactions.get(*args, **kwargs))

    def run_streaming_fetch_with_caller_error(
        self, *args: Any, **kwargs: Any
    ) -> None:
        with self.client.interactions.get(*args, **kwargs):
            raise RuntimeError("caller blew up")

    def drain_stream(self, *args: Any, **kwargs: Any) -> Any:
        stream = self.client.interactions.get(*args, **kwargs)
        list(stream)
        return stream

    def run_streaming_fetch_closed_early(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        stream = self.client.interactions.get(*args, **kwargs)
        first = next(iter(stream))
        stream.close()
        return first
