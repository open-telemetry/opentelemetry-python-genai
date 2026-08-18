# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry smolagents Instrumentation
========================================

Instrumentation for `smolagents <https://github.com/huggingface/smolagents>`_.

Calls to the in-process model classes (``TransformersModel``, ``VLLMModel`` and
``MLXModel``) are recorded as ``chat`` spans. The API-backed model classes are
not instrumented here: each one calls a client library that carries its own
instrumentation, and emitting a span at this layer as well would duplicate the
span and count the token-usage and duration metrics twice. Agent runs and tool
calls are not instrumented yet.

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )
    from smolagents import TransformersModel

    SmolagentsInstrumentor().instrument()

    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct")
    model.generate(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "How many seconds are in a week?"}
                ],
            }
        ]
    )

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

from smolagents import models
from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import model_generate, model_generate_stream

__all__ = ["SmolagentsInstrumentor"]


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
