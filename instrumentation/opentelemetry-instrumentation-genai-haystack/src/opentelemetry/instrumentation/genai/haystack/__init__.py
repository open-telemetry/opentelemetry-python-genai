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

- ``Pipeline.run`` / ``Pipeline.run_async`` / ``Pipeline.run_async_generator`` — one ``invoke_workflow`` span per
  pipeline execution.

See ``tests/conformance/`` for the exact operations covered and the
package ``README.rst``'s "Known limitations" section for the full list of
gaps.

API
---
"""

from __future__ import annotations

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .package import _instruments
from .patch import (
    pipeline_run,
    pipeline_run_async,
    pipeline_run_async_generator,
)


class HaystackInstrumentor(BaseInstrumentor):
    """An instrumentor for the Haystack framework."""

    def __init__(self) -> None:
        super().__init__()

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

        from haystack import Pipeline  # pylint: disable=import-outside-toplevel

        wrap_function_wrapper(Pipeline, "run", pipeline_run(handler))
        if hasattr(Pipeline, "run_async"):
            wrap_function_wrapper(
                Pipeline, "run_async", pipeline_run_async(handler)
            )
        if hasattr(Pipeline, "run_async_generator"):
            wrap_function_wrapper(
                Pipeline,
                "run_async_generator",
                pipeline_run_async_generator(handler),
            )

    def _uninstrument(self, **kwargs: Any) -> None:
        from haystack import Pipeline  # pylint: disable=import-outside-toplevel

        unwrap(Pipeline, "run")
        if hasattr(Pipeline, "run_async"):
            unwrap(Pipeline, "run_async")
        if hasattr(Pipeline, "run_async_generator"):
            unwrap(Pipeline, "run_async_generator")
