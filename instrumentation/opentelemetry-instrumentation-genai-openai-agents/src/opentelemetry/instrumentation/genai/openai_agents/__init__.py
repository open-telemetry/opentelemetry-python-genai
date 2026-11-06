# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenTelemetry instrumentation for the ``openai-agents`` SDK.

Target library:
    https://pypi.org/project/openai-agents/

The agents library provides high-level workflow, agent, and tool orchestration
layered on top of the OpenAI API. It ships its own tracing subsystem
(``agents.tracing``) that emits tree-structured traces of spans
(``AgentSpanData``, ``FunctionSpanData``, etc.).

This instrumentation registers a custom ``TracingProcessor`` with that
subsystem and maps the agents library's lifecycle events to semantic-conventions
spans via ``opentelemetry-util-genai``:

* Top-level ``Trace`` objects map to ``invoke_workflow`` spans.
* ``AgentSpanData`` spans map to ``invoke_agent`` (INTERNAL) spans.
* ``FunctionSpanData`` spans map to ``execute_tool`` spans.

LLM calls (e.g. ``GenerationSpanData`` / ``ResponseSpanData``) are intentionally
NOT mapped here — users instrument the underlying client (typically
``opentelemetry-instrumentation-genai-openai``) to capture chat-completions
and embeddings calls.

Usage::

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
from typing import Any

from agents import Runner
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
from .patch import (
    _current_conversation_id,
    _current_session,
    _extract_conversation_id,
    _extract_session,
)
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

    def __init__(self) -> None:
        super().__init__()

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if self._processor is not None:
            return

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
            bound.arguments["hooks"] = GenAIRunHooks(hooks, processor)

        conv_id = _extract_conversation_id(args, kwargs)
        session = _extract_session(args, kwargs)

        if inspect.iscoroutinefunction(wrapped):

            async def _run_async() -> Any:
                token_conv = _current_conversation_id.set(conv_id)
                token_sess = _current_session.set(session)
                try:
                    return await wrapped(*bound.args, **bound.kwargs)
                finally:
                    _current_conversation_id.reset(token_conv)
                    _current_session.reset(token_sess)

            return _run_async()

        token_conv = _current_conversation_id.set(conv_id)
        token_sess = _current_session.set(session)
        try:
            return wrapped(*bound.args, **bound.kwargs)
        finally:
            _current_conversation_id.reset(token_conv)
            _current_session.reset(token_sess)

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
            self._processor.shutdown()
        finally:
            self._processor = None
            self._previous_processors = None
            self._original_runner_methods = {}
