# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Cohere Instrumentation
====================================

Instrumentation for the `Cohere Python SDK
<https://github.com/cohere-ai/cohere-python>`_.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.cohere import CohereInstrumentor
    import cohere

    # Enable instrumentation
    CohereInstrumentor().instrument()

    # Use the Cohere V2 client normally
    co = cohere.ClientV2()
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "hello world!"}],
    )
    print(response)

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.cohere.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler


class CohereInstrumentor(BaseInstrumentor):
    """An instrumentor for the Cohere Python SDK."""

    def __init__(self) -> None:
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Cohere instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        # Patching will be added in a follow-up PR
        self._handler = handler

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Cohere instrumentation."""
        # Unpatching will be added in a follow-up PR
