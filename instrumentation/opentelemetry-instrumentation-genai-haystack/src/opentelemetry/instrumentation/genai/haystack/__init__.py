# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry Haystack Instrumentation
=======================================

Instrumentation for the `Haystack <https://haystack.deepset.ai/>`_ Python
framework.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor
    from haystack import Pipeline
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    HaystackInstrumentor().instrument()

    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o"))
    pipeline.run({"llm": {"messages": [ChatMessage.from_user("Hello!")]}})

What gets instrumented
-----------------------

- ``Pipeline.run`` / ``Pipeline.run_async`` — one ``invoke_workflow`` span per
  pipeline execution.
- Components classified as a generator (``chat`` / ``text_completion``), an
  embedder (``embeddings``), or a retriever/ranker (``retrieval``) — one span
  per component ``run`` / ``run_async`` call, classified by class name and
  ``run`` method type hints (Haystack has no static component-kind marker).
  Components that don't fall into one of these (prompt builders, routers,
  converters, agents, tool invokers, ...) are not wrapped — there is no
  corresponding ``opentelemetry-util-genai`` invocation type for a generic
  pipeline step. See ``tests/conformance/`` and this package's
  ``MIGRATION_REPORT.md`` for the full list of gaps.

Configuration
-------------

Message content capture can be enabled by setting the environment variable:
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true``

API
---
"""

from __future__ import annotations

from typing import Any, Callable, Collection, Dict, Optional, Type

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .component_types import ComponentType, get_component_type
from .package import _instruments
from .patch import (
    component_run,
    component_run_async,
    pipeline_run,
    pipeline_run_async,
)


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: Optional[TelemetryHandler] = None
        self._original_pipeline_run: Optional[Callable[..., Any]] = None
        self._original_pipeline_run_async: Optional[Callable[..., Any]] = None
        self._original_run_component: Optional[Callable[..., Any]] = None
        self._original_run_component_async: Optional[Callable[..., Any]] = None
        self._wrapped_component_classes: Dict[Type[Any], ComponentType] = {}

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
        self._handler = handler

        import haystack  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
        from haystack.core.component.component import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            component,
        )

        self._original_pipeline_run = haystack.Pipeline.run
        wrap_function_wrapper(haystack.Pipeline, "run", pipeline_run(handler))

        self._original_pipeline_run_async = haystack.Pipeline.run_async
        wrap_function_wrapper(
            haystack.Pipeline, "run_async", pipeline_run_async(handler)
        )

        # Eagerly wrap every classified component already registered (i.e.
        # already imported) at instrumentation time.
        for _class_path, component_cls in list(component.registry.items()):
            self._wrap_component_class(component_cls)

        # Components are frequently imported (and so registered) *after*
        # `instrument()` runs. `Pipeline._run_component[_async]` is called
        # for every component on every pipeline run, so hook it to lazily
        # wrap any component class seen for the first time.
        self._original_run_component = haystack.Pipeline.__dict__[
            "_run_component"
        ].__func__
        wrap_function_wrapper(
            haystack.Pipeline,
            "_run_component",
            self._make_lazy_wrap_wrapper(is_async=False),
        )
        self._original_run_component_async = haystack.Pipeline.__dict__[
            "_run_component_async"
        ].__func__
        wrap_function_wrapper(
            haystack.Pipeline,
            "_run_component_async",
            self._make_lazy_wrap_wrapper(is_async=True),
        )

    def _wrap_component_class(self, component_cls: Type[Any]) -> None:
        if component_cls in self._wrapped_component_classes:
            return
        component_type = get_component_type(component_cls)
        self._wrapped_component_classes[component_cls] = component_type
        if component_type is ComponentType.UNKNOWN:
            return
        handler = self._handler
        assert handler is not None
        if callable(getattr(component_cls, "run", None)):
            wrap_function_wrapper(
                component_cls, "run", component_run(handler, component_type)
            )
        if callable(getattr(component_cls, "run_async", None)):
            wrap_function_wrapper(
                component_cls,
                "run_async",
                component_run_async(handler, component_type),
            )

    def _make_lazy_wrap_wrapper(self, *, is_async: bool) -> Callable[..., Any]:
        def sync_wrapper(
            wrapped: Callable[..., Any],
            instance: Any,  # noqa: ARG001
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            self._maybe_wrap_from_run_component_call(args, kwargs)
            return wrapped(*args, **kwargs)

        async def async_wrapper(
            wrapped: Callable[..., Any],
            instance: Any,  # noqa: ARG001
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            self._maybe_wrap_from_run_component_call(args, kwargs)
            return await wrapped(*args, **kwargs)

        return async_wrapper if is_async else sync_wrapper

    def _maybe_wrap_from_run_component_call(
        self, args: tuple[Any, ...], kwargs: Dict[str, Any]
    ) -> None:
        # `_run_component[_async](component_name, component, ...)` -- `component` is
        # positional arg index 1, a dict with an `"instance"` key.
        component_arg = kwargs.get("component")
        if component_arg is None and len(args) > 1:
            component_arg = args[1]
        if not isinstance(component_arg, dict):
            return
        instance = component_arg.get("instance")
        if instance is None:
            return
        self._wrap_component_class(instance.__class__)

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        if self._original_pipeline_run is not None:
            unwrap(haystack.Pipeline, "run")
        if self._original_pipeline_run_async is not None:
            unwrap(haystack.Pipeline, "run_async")
        if self._original_run_component is not None:
            setattr(
                haystack.Pipeline,
                "_run_component",
                staticmethod(self._original_run_component),
            )
        if self._original_run_component_async is not None:
            setattr(
                haystack.Pipeline,
                "_run_component_async",
                staticmethod(self._original_run_component_async),
            )
        for (
            component_cls,
            component_type,
        ) in self._wrapped_component_classes.items():
            if component_type is ComponentType.UNKNOWN:
                continue
            if callable(getattr(component_cls, "run", None)):
                unwrap(component_cls, "run")
            if callable(getattr(component_cls, "run_async", None)):
                unwrap(component_cls, "run_async")
        self._wrapped_component_classes.clear()
        self._handler = None
