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
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, and ``SPAN_AND_EVENT``.

Captured content can be forwarded to external storage with a completion hook.
Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` (with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``), or pass one programmatically
via ``instrument(completion_hook=...)`` which takes precedence over the
environment variable.

API
---
"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

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
    structured_tool_invoke,
    tool_run,
)

__all__ = ["CrewAIInstrumentor"]


class CrewAIInstrumentor(BaseInstrumentor):
    """An instrumentor for CrewAI."""

    _wrapped: list[tuple[type, str]] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable CrewAI instrumentation."""
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
        from crewai.agent.core import Agent
        from crewai.tools.base_tool import BaseTool
        from crewai.tools.structured_tool import CrewStructuredTool

        self._wrapped = []
        try:
            for cls, method, factory in (
                (Agent, "execute_task", agent_execute_task),
                (Agent, "kickoff", agent_kickoff),
                (BaseTool, "run", tool_run),
                (
                    CrewStructuredTool,
                    "invoke",
                    structured_tool_invoke,
                ),
            ):
                wrap_function_wrapper(cls, method, factory(handler))
                self._wrapped.append((cls, method))
        except BaseException:
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable CrewAI instrumentation."""
        for cls, method in reversed(self._wrapped):
            unwrap(cls, method)
        self._wrapped = []
