# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry Amazon Bedrock instrumentation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import patch_bedrock, unpatch_bedrock

__all__ = ["BedrockInstrumentor"]


class BedrockInstrumentor(BaseInstrumentor):
    """An instrumentor for Amazon Bedrock."""

    _handler: TelemetryHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Amazon Bedrock instrumentation."""
        completion_hook = (
            kwargs.get("completion_hook") or load_completion_hook()
        )
        self._handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=completion_hook,
        )
        patch_bedrock(self._handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Amazon Bedrock instrumentation."""
        unpatch_bedrock()
        self._handler = None
