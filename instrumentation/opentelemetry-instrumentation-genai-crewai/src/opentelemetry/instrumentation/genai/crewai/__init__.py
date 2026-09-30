# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry CrewAI Instrumentation
====================================

Instrumentation for `CrewAI <https://github.com/crewAIInc/crewAI>`_.

``Agent.execute_task``, ``Agent.aexecute_task``, ``Agent.kickoff``, and
``Agent.kickoff_async`` calls are recorded as ``invoke_agent`` spans. Tool
executions that CrewAI runs itself
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

CrewAI's built-in telemetry is left unchanged. Set
``CREWAI_DISABLE_TELEMETRY=true`` before importing CrewAI to opt out using
CrewAI's own configuration. On releases before 1.15 this is also necessary
when configuring OpenTelemetry through the global ``TracerProvider`` because
CrewAI installs its provider during import. Explicit providers passed to
``instrument()`` are unaffected.

API
---
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from crewai.agent.core import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.experimental.agent_executor import AgentExecutor
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
    agent_aexecute_task,
    agent_execute_task,
    agent_kickoff,
    agent_kickoff_async,
    native_tool_call_context,
    tool_execution,
)

__all__ = ["CrewAIInstrumentor"]


class CrewAIInstrumentor(BaseInstrumentor):
    """An instrumentor for CrewAI.

    Patches the synchronous and asynchronous Agent execution APIs,
    ``BaseTool.run``, and ``CrewStructuredTool.invoke`` so that agent
    invocations and tool executions are reported through
    ``opentelemetry-util-genai`` as ``invoke_agent`` and ``execute_tool``
    spans and duration metrics.

    The constructor takes no arguments; behavior is configured through the
    keyword arguments of ``instrument()``:

    ``tracer_provider`` / ``meter_provider`` / ``logger_provider``
        Providers used to emit telemetry. Each defaults to the global
        provider.
    ``completion_hook``
        A ``CompletionHook`` that receives captured prompt and completion
        content. Defaults to the hook selected by
        ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``, or a no-op.
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
                (Agent, "aexecute_task", agent_aexecute_task),
                (Agent, "kickoff", agent_kickoff),
                (Agent, "kickoff_async", agent_kickoff_async),
                (BaseTool, "run", tool_execution),
                (CrewStructuredTool, "invoke", tool_execution),
            ):
                wrap_function_wrapper(cls, method, factory(handler))
                self._wrapped.append((cls, method))
            for cls in (AgentExecutor, CrewAgentExecutor):
                method = "_execute_single_native_tool_call"
                wrap_function_wrapper(cls, method, native_tool_call_context())
                self._wrapped.append((cls, method))
        except BaseException:
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable CrewAI instrumentation.

        Removes every wrapper installed by ``_instrument()`` in reverse
        installation order, restoring the original CrewAI methods.

        Args:
            **kwargs: Ignored; accepted for ``BaseInstrumentor`` compatibility.
        """
        for cls, method in reversed(self._wrapped):
            unwrap(cls, method)
        self._wrapped = []
