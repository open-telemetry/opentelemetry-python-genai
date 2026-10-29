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

SDK lifecycle hooks populate each agent invocation's client-side tool call
counter. Inference calls are counted at the retry helper's model-attempt
boundary when the SDK supports runner-managed retries, otherwise at model-start
hooks. ``opentelemetry-util-genai`` records these as histogram metrics, not span
attributes. Counters remain unset for SDK spans that end before any agent-start
or call-start hook is observed.

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

import inspect
import logging
from collections.abc import Callable, Collection
from importlib import import_module
from typing import Any

from agents import ModelSettings, Runner
from agents.tracing import (
    add_trace_processor,
    get_trace_provider,
    set_trace_processors,
)
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.openai_agents.version import (
    __version__,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAI,
)
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .processor import GenAIRunHooks, GenAITracingProcessor

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

    # BaseInstrumentor is a singleton: __init__ re-runs on the existing
    # instance, so state needed for uninstrumentation lives on the class.
    _processor: GenAITracingProcessor | None = None
    _previous_processors: tuple[Any, ...] | None = None
    _original_runner_methods: dict[str, Any] = {}
    _original_retry_functions: dict[str, Callable[..., Any]] = {}

    def __init__(self) -> None:
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if self._processor is not None:
            return

        if hasattr(ModelSettings, "retry"):
            retry_module = import_module("agents.run_internal.run_loop")
            self._original_retry_functions = {
                name: getattr(retry_module, name)
                for name in (
                    "get_response_with_retry",
                    "stream_response_with_retry",
                )
            }

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

        self._original_runner_methods = {
            name: Runner.__dict__[name]
            for name in ("run", "run_sync", "run_streamed")
        }
        for name in self._original_runner_methods:
            wrap_function_wrapper(
                "agents", f"Runner.{name}", self._wrap_runner
            )
        for name in self._original_retry_functions:
            wrap_function_wrapper(
                "agents.run_internal.run_loop", name, self._wrap_model_retry
            )

    def _wrap_runner(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        bound = inspect.signature(wrapped).bind(*args, **kwargs)
        processor = self._processor
        hooks = bound.arguments.get("hooks")
        if processor is not None and not isinstance(hooks, GenAIRunHooks):
            bound.arguments["hooks"] = GenAIRunHooks(
                hooks,
                processor,
                count_inference_calls=not self._original_retry_functions,
            )
        return wrapped(*bound.args, **bound.kwargs)

    def _wrap_model_retry(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        bound = inspect.signature(wrapped).bind(*args, **kwargs)
        argument = (
            "get_response"
            if "get_response" in bound.arguments
            else "get_stream"
        )
        model_call = bound.arguments[argument]
        processor = self._processor

        def model_attempt() -> Any:
            if processor is not None:
                invocation = processor._get_current_agent()
                if invocation is not None:
                    invocation.inference_calls = (
                        invocation.inference_calls or 0
                    ) + 1
            return model_call()

        bound.arguments[argument] = model_attempt
        return wrapped(*bound.args, **bound.kwargs)

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
            for name, method in self._original_runner_methods.items():
                setattr(Runner, name, method)
            if self._original_retry_functions:
                retry_module = import_module("agents.run_internal.run_loop")
                for name, function in self._original_retry_functions.items():
                    setattr(retry_module, name, function)
            self._processor.shutdown()
        finally:
            self._processor = None
            self._previous_processors = None
            self._original_runner_methods = {}
            self._original_retry_functions = {}
