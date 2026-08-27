# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry Amazon Bedrock instrumentation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.bedrock.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

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
        # Amazon Bedrock patching will be added in a follow-up change.

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Amazon Bedrock instrumentation."""
        self._handler = None
        # Amazon Bedrock unpatching will be added in a follow-up change.
