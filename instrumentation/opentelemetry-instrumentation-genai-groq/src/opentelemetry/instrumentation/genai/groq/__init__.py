# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Groq client instrumentation supporting `groq`_, it can be enabled by
using ``GroqInstrumentor``.

.. _groq: https://pypi.org/project/groq/

Usage
-----

.. code:: python

    from groq import Groq
    from opentelemetry.instrumentation.genai.groq import GroqInstrumentor

    GroqInstrumentor().instrument()

    client = Groq()
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )

Configuration
-------------

This instrumentation emits telemetry using the latest GenAI semantic
conventions and does not capture prompt or completion content by default.
Behavior is controlled via environment variables:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` - enable capture of
    prompts, completions, tool arguments, and return values. Supported values
    are ``span_only``, ``event_only``, and ``span_and_event``.
- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` together with
  ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=<fsspec-uri>`` - upload
  prompts and completions to an ``fsspec``-compatible destination
  (local filesystem, ``gs://``, ``s3://``, etc.) and record reference URIs as
  ``gen_ai.input.messages.ref`` / ``gen_ai.output.messages.ref`` attributes.
  Inline content is not captured unless
  ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is also set.

See the `opentelemetry-util-genai README
<https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/util/opentelemetry-util-genai/README.rst>`_
for the full list of GenAI configuration variables.

A custom ``CompletionHook`` implementation can also be passed programmatically::

    GroqInstrumentor().instrument(completion_hook=my_hook)

When provided, this takes precedence over the hook resolved from
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``.

API
---
"""

from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.groq.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)

from .patch import (
    async_chat_completions_create_v_new,
    chat_completions_create_v_new,
)


def _is_parse_supported():
    """Check if the parse() method is available on the Completions class."""
    try:
        from groq.resources.chat.completions import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            Completions,
        )

        return hasattr(Completions, "parse")
    except ImportError:
        return False


class GroqInstrumentor(BaseInstrumentor):
    def __init__(self):
        self._parse_supported = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Enable Groq instrumentation."""

        tracer_provider = kwargs.get("tracer_provider")
        logger_provider = kwargs.get("logger_provider")
        meter_provider = kwargs.get("meter_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        wrap_function_wrapper(
            "groq.resources.chat.completions",
            "Completions.create",
            chat_completions_create_v_new(handler),
        )

        wrap_function_wrapper(
            "groq.resources.chat.completions",
            "AsyncCompletions.create",
            async_chat_completions_create_v_new(handler),
        )

        self._parse_supported = _is_parse_supported()
        if self._parse_supported:
            wrap_function_wrapper(
                "groq.resources.chat.completions",
                "Completions.parse",
                chat_completions_create_v_new(handler),
            )

            wrap_function_wrapper(
                "groq.resources.chat.completions",
                "AsyncCompletions.parse",
                async_chat_completions_create_v_new(handler),
            )

    def _uninstrument(self, **kwargs):
        import groq  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(groq.resources.chat.completions.Completions, "create")
        unwrap(groq.resources.chat.completions.AsyncCompletions, "create")
        if self._parse_supported:
            unwrap(groq.resources.chat.completions.Completions, "parse")
            unwrap(groq.resources.chat.completions.AsyncCompletions, "parse")
