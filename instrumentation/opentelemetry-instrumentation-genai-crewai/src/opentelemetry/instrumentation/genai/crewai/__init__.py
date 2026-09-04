# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry CrewAI instrumentation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.crewai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["CrewAIInstrumentor"]


class CrewAIInstrumentor(BaseInstrumentor):
    """An instrumentor for CrewAI."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable CrewAI instrumentation."""
        completion_hook = (
            kwargs.get("completion_hook") or load_completion_hook()
        )
        telemetry_handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=completion_hook,
        )
        from opentelemetry.instrumentation.genai.crewai.event_listener import (
            CrewAIEventListener,
        )

        self._event_listener = CrewAIEventListener(telemetry_handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable CrewAI instrumentation."""
        listener = getattr(self, "_event_listener", None)
        if listener is not None:
            listener.shutdown()
            self._event_listener = None
