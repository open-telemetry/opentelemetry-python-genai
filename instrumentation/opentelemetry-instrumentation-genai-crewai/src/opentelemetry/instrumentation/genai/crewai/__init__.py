# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry CrewAI Instrumentation
====================================

Instrumentation for `CrewAI <https://github.com/crewAIInc/crewAI>`_.

Synchronous ``Agent.execute_task`` and standalone ``Agent.kickoff`` calls are
recorded as ``invoke_agent`` spans. Tool executions that CrewAI runs itself
(``BaseTool.run`` on the direct and native function-calling paths, and
``CrewStructuredTool.invoke`` on the ReAct path) are recorded as
``execute_tool`` spans, parented to the agent invocation that triggered them.

Model calls are not instrumented here: CrewAI delegates them to a provider
client library (or LiteLLM) that carries its own instrumentation, and emitting
a span at this layer as well would duplicate the span and count the token-usage
and duration metrics twice.

Usage
-----

.. code-block:: python

    from crewai import Agent, Task

    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()

    agent = Agent(
        role="Writer",
        goal="Write concise answers",
        backstory="An example CrewAI agent",
    )
    task = Task(
        description="Explain OpenTelemetry in one sentence.",
        expected_output="One sentence",
        agent=agent,
    )
    agent.execute_task(task)

Configuration
-------------

Message content capture can be configured by setting the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to ``SPAN_ONLY`` or
``SPAN_AND_EVENT``. Agent and tool invocations record content as span
attributes only and emit no event logs, so ``EVENT_ONLY`` captures nothing.

Captured content can be forwarded to external storage with a completion hook.
Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` (with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``), or pass one programmatically
via ``instrument(completion_hook=...)`` which takes precedence over the
environment variable.

CrewAI ships anonymous usage telemetry that exports spans to
``telemetry.crewai.com``. It is turned off while this instrumentation is
active; pass ``instrument(disable_crewai_telemetry=False)`` to keep it. CrewAI
releases before 1.15 also install that telemetry's ``TracerProvider`` as the
global provider at ``import crewai``, which ``instrument()`` cannot undo — set
``CREWAI_DISABLE_TELEMETRY=true`` before importing CrewAI (or this package,
which imports it) on those releases.
CrewAI's opt-in cloud tracing (``CREWAI_TRACING_ENABLED``) is left untouched.

API
---
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any

from crewai.agent.core import Agent
from crewai.telemetry.telemetry import Telemetry
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.crewai.package import _instruments
from opentelemetry.instrumentation.genai.crewai.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .patch import (
    agent_execute_task,
    agent_kickoff,
    crewai_telemetry_disabled,
    tool_execution,
)

__all__ = ["CrewAIInstrumentor"]

_logger = logging.getLogger(__name__)

_TELEMETRY_CHOKE_POINT = "_safe_telemetry_operation"


class CrewAIInstrumentor(BaseInstrumentor):
    """An instrumentor for CrewAI.

    Patches ``Agent.execute_task``, ``Agent.kickoff``, ``BaseTool.run`` and
    ``CrewStructuredTool.invoke`` so that agent invocations and tool
    executions are reported through ``opentelemetry-util-genai`` as
    ``invoke_agent`` and ``execute_tool`` spans and duration metrics.

    The constructor takes no arguments; behavior is configured through the
    keyword arguments of ``instrument()``:

    ``tracer_provider`` / ``meter_provider`` / ``logger_provider``
        Providers used to emit telemetry. Each defaults to the global
        provider.
    ``completion_hook``
        A ``CompletionHook`` that receives captured prompt and completion
        content. Defaults to the hook selected by
        ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``, or a no-op.
    ``disable_crewai_telemetry`` (default ``True``)
        When ``True``, CrewAI's built-in usage telemetry (spans exported to
        ``telemetry.crewai.com``) is suppressed while the instrumentor is
        active and restored on ``uninstrument()``. When ``False``, it keeps
        running alongside the OpenTelemetry emission.

    Example:
        Route CrewAI telemetry to an explicit tracer provider::

            from opentelemetry.sdk.trace import TracerProvider

            CrewAIInstrumentor().instrument(tracer_provider=TracerProvider())
    """

    _wrapped: list[tuple[type, str]] = []
    """Patched ``(class, method name)`` pairs, in installation order.

    Kept on the class rather than the instance: ``BaseInstrumentor`` is a
    per-class singleton whose ``__init__`` runs on every construction, so
    ``CrewAIInstrumentor().uninstrument()`` would otherwise see empty state.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return the library requirements this instrumentor supports.

        Returns:
            The ``crewai`` version specifiers that must be satisfied for
            ``instrument()`` to patch the library.
        """
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable CrewAI instrumentation.

        Builds the shared ``TelemetryHandler`` and installs the wrappers from
        the ``patch`` module. If any wrapper fails to install, the ones already
        installed are removed before the error propagates, because
        ``BaseInstrumentor`` would not otherwise allow ``uninstrument()`` to
        run.

        Args:
            **kwargs: Optional keyword arguments forwarded from
                ``instrument()``:

                - ``tracer_provider``: ``TracerProvider`` to emit spans with.
                - ``meter_provider``: ``MeterProvider`` to emit metrics with.
                - ``logger_provider``: ``LoggerProvider`` to emit logs with.
                - ``completion_hook``: ``CompletionHook`` for captured content.
                - ``disable_crewai_telemetry``: whether to suppress CrewAI's
                  built-in usage telemetry (default ``True``).

        Raises:
            BaseException: Re-raised unchanged from a failed patch after the
                partially installed wrappers have been removed.
        """
        completion_hook = (
            kwargs.get("completion_hook") or load_completion_hook()
        )
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=completion_hook,
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )
        self._wrapped = []
        try:
            for cls, method, factory in (
                (Agent, "execute_task", agent_execute_task),
                (Agent, "kickoff", agent_kickoff),
                (BaseTool, "run", tool_execution),
                (CrewStructuredTool, "invoke", tool_execution),
            ):
                wrap_function_wrapper(cls, method, factory(handler))
                self._wrapped.append((cls, method))
            if kwargs.get("disable_crewai_telemetry", True):
                self._disable_crewai_telemetry()
        except BaseException:
            self._uninstrument()
            raise

    def _disable_crewai_telemetry(self) -> None:
        """Suppress CrewAI's built-in usage telemetry.

        Wraps the private method every CrewAI telemetry emitter funnels
        through so that no span is created or exported. The wrapper is
        registered alongside the other patches and removed on
        ``_uninstrument()``. If CrewAI has renamed that method, a warning is
        logged and instrumentation continues without suppression.
        """
        # The choke point is private to CrewAI, so a rename must not break
        # instrument(); the user can still fall back to the env var.
        if getattr(Telemetry, _TELEMETRY_CHOKE_POINT, None) is None:
            _logger.warning(
                "Could not disable CrewAI usage telemetry: "
                "Telemetry.%s not found. Set CREWAI_DISABLE_TELEMETRY=true "
                "to disable it.",
                _TELEMETRY_CHOKE_POINT,
            )
            return
        wrap_function_wrapper(
            Telemetry, _TELEMETRY_CHOKE_POINT, crewai_telemetry_disabled
        )
        self._wrapped.append((Telemetry, _TELEMETRY_CHOKE_POINT))

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable CrewAI instrumentation.

        Removes every wrapper installed by ``_instrument()`` in reverse
        installation order, restoring the original CrewAI methods and its
        built-in telemetry.

        Args:
            **kwargs: Ignored; accepted for ``BaseInstrumentor`` compatibility.
        """
        for cls, method in reversed(self._wrapped):
            unwrap(cls, method)
        self._wrapped = []
