# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry smolagents Instrumentation
========================================

Instrumentation for `smolagents <https://github.com/huggingface/smolagents>`_.

Agent runs are recorded as ``invoke_agent`` spans and tool calls as
``execute_tool`` spans.

Calls to the in-process model classes (``TransformersModel``, ``VLLMModel`` and
``MLXModel``) are recorded as ``chat`` spans. The API-backed model classes are
not instrumented here: each one calls a client library that carries its own
instrumentation, and emitting a span at this layer as well would duplicate the
span and count the token-usage and duration metrics twice. Install the client
library's own instrumentation to record those model calls.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )
    from smolagents import CodeAgent, TransformersModel

    SmolagentsInstrumentor().instrument()

    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct")
    agent = CodeAgent(tools=[], model=model)
    agent.run("How many seconds are in a week?")

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

from collections.abc import Callable, Collection
from contextvars import copy_context
from functools import wraps
from typing import Any

import smolagents
from smolagents import models
from smolagents.tools import Tool
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import (
    agent_run,
    agent_tool_calls,
    model_generate,
    model_generate_stream,
    tool_call,
)

__all__ = ["SmolagentsInstrumentor"]

_LOCAL_EXECUTOR_MODULE = "smolagents.local_python_executor"

# The model classes that run inference in the current process. They call no
# client library, so this instrumentation is the only place their model calls
# can be observed.
#
# The API-backed classes are left out on purpose. Each one calls a client
# library whose own instrumentation emits the ``chat`` span, so wrapping them
# here as well would produce two spans for one model call and count the
# token-usage and duration metrics twice. ``README.rst`` lists which
# instrumentation covers which class.
_IN_PROCESS_MODEL_CLASSES = ("MLXModel", "TransformersModel", "VLLMModel")


def _model_classes_defining(method: str) -> list[type]:
    """The in-process model classes whose ``method`` gets wrapped.

    Only classes that define ``method`` in their own ``__dict__`` are patched.
    ``MLXModel`` and ``VLLMModel`` have no ``generate_stream``, and neither does
    the base class, so wrapping it on them raises ``AttributeError``. The same
    check keeps a method defined on a shared base from being wrapped once per
    subclass.

    A user-defined subclass that overrides the method shadows the patched one and
    emits no ``chat`` span. ``README.rst`` documents that limitation.

    A class is looked up by name so that a smolagents version without one of them
    is skipped rather than raising.
    """
    classes: list[type] = []
    for name in _IN_PROCESS_MODEL_CLASSES:
        model_cls = getattr(models, name, None)
        if isinstance(model_cls, type) and method in model_cls.__dict__:
            classes.append(model_cls)
    return classes


def _tool_classes_defining_call() -> list[type]:
    """The tool classes whose ``__call__`` gets wrapped.

    ``PipelineTool`` overrides ``Tool.__call__`` without delegating to it, so
    patching ``Tool`` alone would miss it and the shipped ``SpeechToTextTool``.
    Only classes that define ``__call__`` are patched, so a call emits exactly
    one span. smolagents doesn't export ``PipelineTool``, hence the MRO walk.
    """
    classes: dict[type, None] = {}
    for obj in vars(smolagents).values():
        if not isinstance(obj, type) or not issubclass(obj, Tool):
            continue
        for cls in obj.__mro__:
            if issubclass(cls, Tool) and "__call__" in cls.__dict__:
                classes.setdefault(cls, None)
    return list(classes)


def _context_preserving_timeout(
    wrapped: Callable[..., Any],
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Run code decorated with ``timeout()`` in a copy of the caller's context.

    ``local_python_executor.timeout()`` runs the decorated function in a
    ``ThreadPoolExecutor`` worker and submits it without propagating
    ``contextvars``, so a span started while executing agent-generated code (a
    tool call) loses the active OTel context and becomes a root span instead of
    a child of the agent span. smolagents copies the context for its parallel
    tool calls (``agents.py``: ``ctx = copy_context(); executor.submit(ctx.run,
    ...)``) but not here.

    A ``CodeAgent`` with the default ``executor_type="local"`` goes through this
    on every step; the remote executors do not.
    """
    decorator = wrapped(*args, **kwargs)

    def context_preserving_decorator(
        func: Callable[..., Any],
    ) -> Callable[..., Any]:
        @wraps(func)
        def run_with_caller_context(
            *call_args: Any, **call_kwargs: Any
        ) -> Any:
            context = copy_context()

            def in_caller_context(
                *inner_args: Any, **inner_kwargs: Any
            ) -> Any:
                return context.run(func, *inner_args, **inner_kwargs)

            return decorator(in_caller_context)(*call_args, **call_kwargs)

        return run_with_caller_context

    return context_preserving_decorator


class SmolagentsInstrumentor(BaseInstrumentor):
    """An instrumentor for smolagents."""

    # ``BaseInstrumentor.__new__`` returns a per-class singleton, but Python
    # still runs ``__init__`` on every construction. Initializing this state in
    # ``__init__`` would let the documented ``SmolagentsInstrumentor()
    # .uninstrument()`` form wipe the live instance's bookkeeping and leave
    # smolagents permanently patched, so these are class-level defaults that
    # only ``_instrument`` / ``_uninstrument`` rebind.
    _wrapped_generate_classes: list[type] = []
    _wrapped_generate_stream_classes: list[type] = []
    _wrapped_tool_classes: list[type] = []

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable smolagents instrumentation.

        Args:
            **kwargs: Optional arguments
                - tracer_provider: TracerProvider instance
                - meter_provider: MeterProvider instance
                - logger_provider: LoggerProvider instance
                - completion_hook: CompletionHook instance
        """
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        self._wrapped_generate_classes = []
        self._wrapped_generate_stream_classes = []
        self._wrapped_tool_classes = []
        try:
            for model_cls in _model_classes_defining("generate"):
                wrap_function_wrapper(
                    model_cls,
                    "generate",
                    model_generate(handler),
                )
                self._wrapped_generate_classes.append(model_cls)

            for model_cls in _model_classes_defining("generate_stream"):
                wrap_function_wrapper(
                    model_cls,
                    "generate_stream",
                    model_generate_stream(handler),
                )
                self._wrapped_generate_stream_classes.append(model_cls)

            wrap_function_wrapper(
                "smolagents",
                "MultiStepAgent.run",
                agent_run(handler),
            )
            # Emits no span of its own; it carries the provider's tool call id
            # down to the execute_tool spans of the step.
            wrap_function_wrapper(
                "smolagents",
                "ToolCallingAgent.process_tool_calls",
                agent_tool_calls,
            )

            for tool_cls in _tool_classes_defining_call():
                wrap_function_wrapper(
                    tool_cls,
                    "__call__",
                    tool_call(handler),
                )
                self._wrapped_tool_classes.append(tool_cls)

            wrap_function_wrapper(
                _LOCAL_EXECUTOR_MODULE,
                "timeout",
                _context_preserving_timeout,
            )

            # TODO: emit a span per agent step once the semantic conventions
            # define an operation for one iteration of a reason-and-act loop.
            # Until then execute_tool spans nest directly under invoke_agent.
            # https://github.com/open-telemetry/semantic-conventions-genai/issues/81
        except BaseException:
            # BaseInstrumentor.instrument() doesn't mark the instrumentor as
            # instrumented when _instrument raises, so uninstrument() would
            # refuse to run and leave the patches applied with no way to undo.
            self._uninstrument()
            raise

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable smolagents instrumentation and restore patched originals."""
        for model_cls in self._wrapped_generate_classes:
            unwrap(model_cls, "generate")
        self._wrapped_generate_classes = []

        for model_cls in self._wrapped_generate_stream_classes:
            unwrap(model_cls, "generate_stream")
        self._wrapped_generate_stream_classes = []

        unwrap(smolagents.MultiStepAgent, "run")
        unwrap(smolagents.ToolCallingAgent, "process_tool_calls")

        for tool_cls in self._wrapped_tool_classes:
            unwrap(tool_cls, "__call__")
        self._wrapped_tool_classes = []

        unwrap(_LOCAL_EXECUTOR_MODULE, "timeout")
