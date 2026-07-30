# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared MockModel for Agno tests."""

from __future__ import annotations

from typing import Any

from agno.models.base import Model


class MockModel(Model):
    """Dummy model implementation for testing without provider dependencies."""

    def _parse_provider_response(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def _parse_provider_response_delta(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        pass

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> Any:
        pass
