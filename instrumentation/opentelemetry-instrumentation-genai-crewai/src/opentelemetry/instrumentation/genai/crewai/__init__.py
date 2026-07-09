# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""CrewAI instrumentation using OpenTelemetry GenAI semantic conventions.

This callback-based slice follows the donated OpenInference CrewAI event
listener approach, but emits spans through opentelemetry-util-genai rather than
OpenInference span attributes.
"""

from __future__ import annotations

from typing import Any, Collection

from opentelemetry.instrumentation.genai.crewai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.handler import get_telemetry_handler


class CrewAIInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for CrewAI LLM inference events."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        from opentelemetry.instrumentation.genai.crewai.event_listener import (
            CrewAIInferenceEventListener,
        )

        telemetry_handler = get_telemetry_handler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
        )
        self._event_listener = CrewAIInferenceEventListener(
            telemetry_handler=telemetry_handler
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        if getattr(self, "_event_listener", None) is not None:
            self._event_listener.shutdown()
            self._event_listener = None
        if getattr(get_telemetry_handler, "_default_handler", None) is not None:
            delattr(get_telemetry_handler, "_default_handler")


__all__ = ["CrewAIInstrumentor"]
