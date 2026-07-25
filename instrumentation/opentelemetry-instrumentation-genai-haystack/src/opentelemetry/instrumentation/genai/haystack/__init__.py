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
  embedder (``embeddings``), a retriever/ranker (``retrieval``), or an
  ``Agent`` (``invoke_agent``) — one span per component ``run`` /
  ``run_async`` call, classified by class name and ``run`` method type
  hints (Haystack has no static component-kind marker). Components that
  don't fall into one of these (prompt builders, routers, converters, ...)
  are not wrapped — there is no corresponding ``opentelemetry-util-genai``
  invocation type for a generic pipeline step.
- ``haystack.tools.Tool.invoke`` / ``invoke_async`` — one ``execute_tool``
  span per tool call.

See ``tests/conformance/`` and this package's ``MIGRATION_REPORT.md`` for
the full list of gaps.

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
    pipeline_run_async_generator,
    tool_invoke,
    tool_invoke_async,
)


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

    def __init__(self) -> None:
        super().__init__()
        self._handler: Optional[TelemetryHandler] = None
        self._original_pipeline_run: Optional[Callable[..., Any]] = None
        self._original_pipeline_run_async: Optional[Callable[..., Any]] = None
        self._original_pipeline_run_async_generator: Optional[
            Callable[..., Any]
        ] = None
        self._original_component_register: Optional[Callable[..., Any]] = None
        self._original_tool_invoke: Optional[Callable[..., Any]] = None
        self._original_tool_invoke_async: Optional[Callable[..., Any]] = None
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
            _Component,
            component,
        )

        self._original_pipeline_run = haystack.Pipeline.run
        wrap_function_wrapper(haystack.Pipeline, "run", pipeline_run(handler))

        self._original_pipeline_run_async = haystack.Pipeline.run_async
        wrap_function_wrapper(
            haystack.Pipeline, "run_async", pipeline_run_async(handler)
        )

        self._original_pipeline_run_async_generator = (
            haystack.Pipeline.run_async_generator
        )
        wrap_function_wrapper(
            haystack.Pipeline,
            "run_async_generator",
            pipeline_run_async_generator(handler),
        )

        # Eagerly wrap every classified component already registered (i.e.
        # already imported) at instrumentation time.
        for _class_path, component_cls in list(component.registry.items()):
            self._wrap_component_class(component_cls)

        # Components are frequently imported (and so registered) *after*
        # `instrument()` runs -- and not every component is ever run through
        # a Pipeline (e.g. a Haystack `Agent` calls its `chat_generator.run()`
        # directly). `_Component._component` is the single method the
        # `@component` decorator calls to validate and register a class,
        # for every component regardless of how it's later invoked, so hook
        # it to classify and wrap each component class the instant it's
        # defined.
        self._original_component_register = _Component._component
        wrap_function_wrapper(
            _Component, "_component", self._make_registration_wrapper()
        )

        from haystack.tools import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            Tool,
        )

        self._original_tool_invoke = Tool.invoke
        wrap_function_wrapper(Tool, "invoke", tool_invoke(handler))
        self._original_tool_invoke_async = Tool.invoke_async
        wrap_function_wrapper(Tool, "invoke_async", tool_invoke_async(handler))

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

    def _make_registration_wrapper(self) -> Callable[..., Any]:
        def wrapper(
            wrapped: Callable[..., Any],
            instance: Any,  # noqa: ARG001
            args: tuple[Any, ...],
            kwargs: Dict[str, Any],
        ) -> Any:
            # `_Component._component` may rebuild the class it's given
            # (e.g. to attach a generated `__repr__`), so classify and wrap
            # the *returned* class, not the input one.
            registered_cls = wrapped(*args, **kwargs)
            self._wrap_component_class(registered_cls)
            return registered_cls

        return wrapper

    def _uninstrument(self, **kwargs: Any) -> None:
        import haystack  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        if self._original_pipeline_run is not None:
            unwrap(haystack.Pipeline, "run")
        if self._original_pipeline_run_async is not None:
            unwrap(haystack.Pipeline, "run_async")
        if self._original_pipeline_run_async_generator is not None:
            unwrap(haystack.Pipeline, "run_async_generator")
        if self._original_component_register is not None:
            from haystack.core.component.component import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
                _Component,
            )

            unwrap(_Component, "_component")
        if (
            self._original_tool_invoke is not None
            or self._original_tool_invoke_async is not None
        ):
            from haystack.tools import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
                Tool,
            )

            if self._original_tool_invoke is not None:
                unwrap(Tool, "invoke")
            if self._original_tool_invoke_async is not None:
                unwrap(Tool, "invoke_async")
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
