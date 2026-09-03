# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry DSPy instrumentation."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.dspy.package import _instruments
from opentelemetry.instrumentation.genai.dspy.patch import (
    patch_dspy,
    unpatch_dspy,
)
from opentelemetry.instrumentation.genai.dspy.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["DSPyInstrumentor", "__version__"]


class DSPyInstrumentor(BaseInstrumentor):
    """An instrumentor for DSPy."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable DSPy instrumentation."""
        completion_hook = (
            kwargs.get("completion_hook") or load_completion_hook()
        )
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=completion_hook,
        )
        patch_dspy(handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable DSPy instrumentation."""
        unpatch_dspy()
