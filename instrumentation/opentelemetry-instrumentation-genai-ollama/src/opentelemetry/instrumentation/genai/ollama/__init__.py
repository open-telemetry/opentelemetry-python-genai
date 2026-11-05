# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Ollama Instrumentation
====================================

Instrumentation for the `Ollama Python library
<https://github.com/ollama/ollama-python>`_.

Usage
-----

.. code-block:: python

    import ollama

    from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor

    OllamaInstrumentor().instrument()

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

Configuration
-------------

Message content capture can be configured by setting the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``no_content``, ``span_only``, ``event_only``, and ``span_and_event``.

API
---
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from opentelemetry.instrumentation.genai.ollama.package import _instruments
from opentelemetry.instrumentation.genai.ollama.patch import (
    patch_ollama,
    unpatch_ollama,
)
from opentelemetry.instrumentation.genai.ollama.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["OllamaInstrumentor"]


class OllamaInstrumentor(BaseInstrumentor):
    """An instrumentor for the Ollama Python library.

    Instruments ``chat`` and ``embed`` on ``ollama.Client`` and
    ``ollama.AsyncClient``, and the module-level ``ollama.chat`` and
    ``ollama.embed`` functions.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )
        patch_ollama(handler)

    def _uninstrument(self, **kwargs: Any) -> None:
        unpatch_ollama()
