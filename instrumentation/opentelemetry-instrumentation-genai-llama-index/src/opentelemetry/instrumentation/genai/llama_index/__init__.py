# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry LlamaIndex Instrumentation
========================================

Instrument LlamaIndex applications by enabling ``LlamaIndexInstrumentor``.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.llama_index import (
        LlamaIndexInstrumentor,
    )

    LlamaIndexInstrumentor().instrument()

Configuration
-------------

Content capture is controlled through
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, and ``SPAN_AND_EVENT``.
Completion hook configuration is forwarded from
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` or
``instrument(completion_hook=...)``.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Protocol, cast

from llama_index.core import instrumentation
from llama_index.core.instrumentation.span_handlers import BaseSpanHandler

from opentelemetry.instrumentation.genai.llama_index._handler import (
    LlamaIndexSpanHandler,
)
from opentelemetry.instrumentation.genai.llama_index.package import (
    _instruments,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["LlamaIndexInstrumentor"]


class _Dispatcher(Protocol):
    span_handlers: list[BaseSpanHandler[Any]]

    def add_span_handler(self, handler: BaseSpanHandler[Any]) -> None: ...


class LlamaIndexInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for LlamaIndex agents and tools."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: TelemetryHandler | None = None
        self._span_handler: LlamaIndexSpanHandler | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )
        span_handler = LlamaIndexSpanHandler(handler)
        dispatcher = cast(
            _Dispatcher, getattr(instrumentation, "get_dispatcher")()
        )
        dispatcher.add_span_handler(span_handler)
        self._handler = handler
        self._span_handler = span_handler

    def _uninstrument(self, **kwargs: Any) -> None:
        dispatcher = cast(
            _Dispatcher, getattr(instrumentation, "get_dispatcher")()
        )
        if self._span_handler in dispatcher.span_handlers:
            dispatcher.span_handlers.remove(self._span_handler)
        self._span_handler = None
        self._handler = None
