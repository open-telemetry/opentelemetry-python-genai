# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry CrewAI instrumentation."""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.crewai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["CrewAIInstrumentor"]

_CREWAI_DISABLE_TELEMETRY = "CREWAI_DISABLE_TELEMETRY"


class CrewAIInstrumentor(BaseInstrumentor):
    """An instrumentor for CrewAI."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable CrewAI instrumentation."""
        self._disabled_crewai_telemetry = (
            _CREWAI_DISABLE_TELEMETRY not in os.environ
        )
        if self._disabled_crewai_telemetry:
            os.environ[_CREWAI_DISABLE_TELEMETRY] = "true"

        try:
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
                CrewAIInferenceEventListener,
            )

            self._event_listener = CrewAIInferenceEventListener(
                telemetry_handler
            )
        except BaseException:
            self._restore_crewai_telemetry()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable CrewAI instrumentation."""
        listener = getattr(self, "_event_listener", None)
        if listener is not None:
            listener.shutdown()
            self._event_listener = None
        self._restore_crewai_telemetry()

    def _restore_crewai_telemetry(self) -> None:
        if getattr(self, "_disabled_crewai_telemetry", False):
            os.environ.pop(_CREWAI_DISABLE_TELEMETRY, None)
            self._disabled_crewai_telemetry = False
