# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry QwenPaw Instrumentation
=====================================

Instrumentation for `QwenPaw <https://github.com/agentscope-ai/QwenPaw>`_,
a personal assistant application built on AgentScope. QwenPaw was
originally published as ``copaw``, so this package ships two instrumentor
plugins targeting the same ``AgentRunner`` surface:

- :class:`QwenPawInstrumentor` for the ``qwenpaw`` distribution
- :class:`CoPawInstrumentor` for the legacy ``copaw`` distribution

Each user turn handled by ``AgentRunner.query_handler`` is traced as one
``invoke_agent`` span following the OpenTelemetry GenAI semantic
conventions.

QwenPaw delegates model (LLM) and tool execution to AgentScope, so this
package emits no ``chat`` or ``execute_tool`` spans and its conformance
suite covers none — those operations belong to the AgentScope
instrumentation. Pair this package with the instrumentations of AgentScope
and the underlying model libraries so their calls appear as child spans
under ``invoke_agent``.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor

    QwenPawInstrumentor().instrument()
    # ... run the QwenPaw app ...
    QwenPawInstrumentor().uninstrument()

Configuration
-------------

Message content capture is enabled by setting the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``.

Captured content can be uploaded to external storage instead of being
recorded inline by configuring a completion hook, either through
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` or by passing
``instrument(completion_hook=...)``.

API
---
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any, ClassVar, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments_copaw, _instruments_qwenpaw
from .patch import make_query_handler_wrapper

logger = logging.getLogger(__name__)

__all__ = ["CoPawInstrumentor", "QwenPawInstrumentor"]


class _AgentRunnerInstrumentor(BaseInstrumentor):
    """Shared ``AgentRunner.query_handler`` instrumentation.

    Subclasses bind one runtime distribution (``qwenpaw`` or legacy
    ``copaw``) via :attr:`_instruments` and :attr:`_runner_module`; both
    expose the same ``AgentRunner`` surface.
    """

    _instruments: ClassVar[tuple[str, ...]]
    _runner_module: ClassVar[str]

    def instrumentation_dependencies(self) -> Collection[str]:
        return self._instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable the ``AgentRunner.query_handler`` instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance, taking precedence
                  over the one configured by environment variable
        """
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )
        wrap_function_wrapper(
            self._runner_module,
            "AgentRunner.query_handler",
            make_query_handler_wrapper(handler),
        )
        logger.debug(
            "Instrumented %s.AgentRunner.query_handler", self._runner_module
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable the ``AgentRunner.query_handler`` instrumentation."""
        del kwargs
        runner_module = import_module(self._runner_module)
        unwrap(runner_module.AgentRunner, "query_handler")
        logger.debug(
            "Uninstrumented %s.AgentRunner.query_handler", self._runner_module
        )


class QwenPawInstrumentor(_AgentRunnerInstrumentor):
    """An instrumentor for the QwenPaw application runner.

    Traces ``AgentRunner.query_handler`` as an ``invoke_agent`` span and
    optionally captures the turn's input/output messages. Model and tool
    calls are delegated to AgentScope and are not instrumented here.
    """

    _instruments = (_instruments_qwenpaw,)
    _runner_module = "qwenpaw.app.runner.runner"


class CoPawInstrumentor(_AgentRunnerInstrumentor):
    """An instrumentor for the legacy ``copaw`` application runner.

    ``copaw`` is QwenPaw's former distribution name; its last release is
    ``copaw 1.0.2``. Apart from the module path the runner surface matches
    QwenPaw's, so the emitted telemetry is identical to
    :class:`QwenPawInstrumentor`'s.
    """

    _instruments = (_instruments_copaw,)
    _runner_module = "copaw.app.runner.runner"
