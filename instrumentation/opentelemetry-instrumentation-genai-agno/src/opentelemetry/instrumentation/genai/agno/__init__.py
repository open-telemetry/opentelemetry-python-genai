# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Agno Instrumentation
==================================

Instrumentation for `Agno <https://github.com/agno-agi/agno>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor

    # Enable instrumentation
    AgnoInstrumentor().instrument()

Configuration
-------------

Message content capture can be configured by setting the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``no_content``, ``span_only``, ``event_only``, and ``span_and_event``.

API
---
"""

from __future__ import annotations

from typing import Any, Collection

from opentelemetry.instrumentation.genai.agno.package import (
    _instruments,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["AgnoInstrumentor"]


class AgnoInstrumentor(BaseInstrumentor):
    """An instrumentor for Agno."""

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Agno instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance
        """
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")
        completion_hook = (
            kwargs.get("completion_hook") or load_completion_hook()
        )

        TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=completion_hook,
        )
        # Patching will be added in a follow-up PR

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Agno instrumentation.

        This removes all patches applied during instrumentation.
        """
        # Unpatching will be added in a follow-up PR
