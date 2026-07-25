# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Agents instrumentation for OpenTelemetry.

Registers a :class:`GenAITracingProcessor` with the agents library's
public ``add_trace_processor`` extension API. The processor reacts
synchronously to the agents library's own ``Trace`` / ``AgentSpan`` /
``FunctionSpan`` start/end callbacks and turns them
into ``invoke_workflow`` / ``invoke_agent`` / ``execute_tool`` spans via
``opentelemetry-util-genai``.

LLM-level spans (``chat`` / ``embeddings``) are produced
by ``opentelemetry-instrumentation-genai-openai`` when both packages are
installed; this instrumentation does not emit them.

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.genai.openai_agents import (
        OpenAIAgentsInstrumentor,
    )

    # Default: keep the OpenAI native trace exporter; add our OTel emission.
    OpenAIAgentsInstrumentor().instrument()

    # Replace the default exporter so traces are only sent via OTel.
    OpenAIAgentsInstrumentor().instrument(disable_openai_trace_export=True)
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from agents.tracing import (
    add_trace_processor,
    get_trace_provider,
    set_trace_processors,
)

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
    get_telemetry_handler,
)

from .package import _instruments
from .processor import GenAITracingProcessor

__all__ = ["OpenAIAgentsInstrumentor"]

logger = logging.getLogger(__name__)


class OpenAIAgentsInstrumentor(BaseInstrumentor):
    """Instrument the openai-agents library.

    Constructor takes no arguments. Configure behavior via ``instrument()``:

    ``disable_openai_trace_export`` (default ``False``)
        When ``False`` (default), the agents library's built-in trace
        exporter (which sends traces to OpenAI's hosted tracing backend
        when ``OPENAI_API_KEY`` is set) remains active alongside our OTel
        emission.

        When ``True``, the default exporter is removed via
        ``agents.tracing.set_trace_processors`` so traces flow only through
        OpenTelemetry while this instrumentor is active. Previously registered
        processors are restored on ``uninstrument()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._processor: GenAITracingProcessor | None = None
        self._previous_processors: tuple[Any, ...] | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if self._processor is not None:
            return

        handler: TelemetryHandler = get_telemetry_handler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )
        provider = GenAI.GenAiProviderNameValues.OPENAI.value
        self._processor = GenAITracingProcessor(handler, provider)

        if kwargs.get("disable_openai_trace_export"):
            trace_provider = get_trace_provider()
            current = getattr(
                getattr(trace_provider, "_multi_processor", None),
                "_processors",
                (),
            )
            self._previous_processors = tuple(current)
            set_trace_processors([self._processor])
        else:
            add_trace_processor(self._processor)

    def _uninstrument(self, **kwargs: Any) -> None:
        if self._processor is None:
            return

        if self._previous_processors is not None:
            set_trace_processors(list(self._previous_processors))
        else:
            provider = get_trace_provider()
            current = getattr(
                getattr(provider, "_multi_processor", None), "_processors", ()
            )
            filtered = [p for p in current if p is not self._processor]
            set_trace_processors(filtered)
        try:
            self._processor.shutdown()
        finally:
            self._processor = None
            self._previous_processors = None
