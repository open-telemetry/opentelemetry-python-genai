# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Qwen-Agent instrumentation supporting the `qwen-agent
<https://pypi.org/project/qwen-agent/>`_ framework, it can be enabled by
using ``QwenAgentInstrumentor``.

Usage
-----
.. code:: python

    from opentelemetry.instrumentation.genai.qwen_agent import (
        QwenAgentInstrumentor,
    )
    from qwen_agent.agents import Assistant

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="my-assistant",
        system_message="You are a helpful assistant.",
    )

    messages = [{"role": "user", "content": "Hello!"}]
    for responses in bot.run(messages):
        pass

    QwenAgentInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

from typing import Any, Callable, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.qwen_agent.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.qwen_agent.patch import (
    wrap_agent_call_tool,
    wrap_agent_run,
)
from opentelemetry.instrumentation.genai.qwen_agent.version import __version__
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["QwenAgentInstrumentor", "__version__"]

_AGENT_MODULE = "qwen_agent.agent"


def _with_handler(
    wrapper: Callable[..., Any], handler: TelemetryHandler
) -> Callable[..., Any]:
    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return wrapper(wrapped, instance, args, kwargs, handler)

    return _wrapper


class QwenAgentInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for the Qwen-Agent framework.

    Instruments the following methods:

    - ``Agent.run()``: agent execution spans (``invoke_agent``).
      ``Agent.run_nonstream()`` is not wrapped separately — it calls
      ``run()`` internally, so the ``invoke_agent`` span is created once
      by the ``run()`` wrapper.
    - ``Agent._call_tool()``: tool execution spans (``execute_tool``).

    LLM call spans (``chat``) are not emitted by this instrumentation;
    enable the instrumentation of the underlying model client library
    (e.g. ``opentelemetry-instrumentation-genai-openai`` for
    OpenAI-compatible backends) to capture them without duplication.
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Qwen-Agent instrumentation."""
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        # Positional arguments for wrapt 1.x/2.x compatibility (wrapt 2.x
        # renamed the first parameter from ``module`` to ``target``).
        wrap_function_wrapper(
            _AGENT_MODULE,
            "Agent.run",
            _with_handler(wrap_agent_run, handler),
        )
        wrap_function_wrapper(
            _AGENT_MODULE,
            "Agent._call_tool",
            _with_handler(wrap_agent_call_tool, handler),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Qwen-Agent instrumentation."""
        import qwen_agent.agent  # noqa: PLC0415

        unwrap(qwen_agent.agent.Agent, "run")
        unwrap(qwen_agent.agent.Agent, "_call_tool")
