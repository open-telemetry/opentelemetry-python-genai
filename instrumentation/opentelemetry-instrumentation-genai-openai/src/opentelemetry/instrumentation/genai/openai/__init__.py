# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI client instrumentation supporting `openai`_, it can be enabled by
using ``OpenAIInstrumentor``.

.. _openai: https://pypi.org/project/openai/

Usage
-----

.. code:: python

    from openai import OpenAI
    from opentelemetry.instrumentation.genai.openai import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    are ``span_only``, ``event_only``, and ``span_and_event``. This requires
    ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental``.
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

    OpenAIInstrumentor().instrument(completion_hook=my_hook)

When provided, this takes precedence over the hook resolved from
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``.

API
---
"""

from importlib import import_module
from typing import Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.openai.package import _instruments
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import (
    TelemetryHandler,
)

from .patch import (
    async_chat_completions_create_v_new,
    async_embeddings_create,
    chat_completions_create_v_new,
    embeddings_create,
)
from .patch_responses import (
    async_responses_create,
    responses_create,
)


def _is_parse_supported():
    """Check if the parse() method is available on the Completions class.

    The parse() method for structured outputs was added in openai >= 1.40.0.
    """
    try:
        from openai.resources.chat.completions import (  # pylint: disable=import-outside-toplevel  # noqa: PLC0415
            Completions,
        )

        return hasattr(Completions, "parse")
    except ImportError:
        return False


class OpenAIInstrumentor(BaseInstrumentor):
    def __init__(self):
        self._parse_supported = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        """Enable OpenAI instrumentation."""

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
            "openai.resources.chat.completions",
            "Completions.create",
            chat_completions_create_v_new(handler),
        )

        wrap_function_wrapper(
            "openai.resources.chat.completions",
            "AsyncCompletions.create",
            async_chat_completions_create_v_new(handler),
        )

        # Add instrumentation for the embeddings API
        wrap_function_wrapper(
            "openai.resources.embeddings",
            "Embeddings.create",
            embeddings_create(handler),
        )

        wrap_function_wrapper(
            "openai.resources.embeddings",
            "AsyncEmbeddings.create",
            async_embeddings_create(handler),
        )

        # parse() wraps create() internally in the OpenAI SDK and returns a
        # ParsedChatCompletion. The telemetry-relevant fields (model, usage,
        # choices, finish_reason) are identical to ChatCompletion, so the
        # existing create() wrappers handle it correctly.
        self._parse_supported = _is_parse_supported()
        if self._parse_supported:
            wrap_function_wrapper(
                "openai.resources.chat.completions",
                "Completions.parse",
                chat_completions_create_v_new(handler),
            )

            wrap_function_wrapper(
                "openai.resources.chat.completions",
                "AsyncCompletions.parse",
                async_chat_completions_create_v_new(handler),
            )

        responses_module = _get_responses_module()
        if responses_module is not None:
            wrap_function_wrapper(
                "openai.resources.responses.responses",
                "Responses.create",
                responses_create(handler),
            )
            wrap_function_wrapper(
                "openai.resources.responses.responses",
                "AsyncResponses.create",
                async_responses_create(handler),
            )

    def _uninstrument(self, **kwargs):
        import openai  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(openai.resources.chat.completions.Completions, "create")
        unwrap(openai.resources.chat.completions.AsyncCompletions, "create")
        unwrap(openai.resources.embeddings.Embeddings, "create")
        unwrap(openai.resources.embeddings.AsyncEmbeddings, "create")
        if self._parse_supported:
            unwrap(openai.resources.chat.completions.Completions, "parse")
            unwrap(openai.resources.chat.completions.AsyncCompletions, "parse")
        responses_module = _get_responses_module()
        if responses_module is not None:
            unwrap(responses_module.Responses, "create")
            if hasattr(responses_module, "AsyncResponses"):
                unwrap(responses_module.AsyncResponses, "create")


def _get_responses_module():
    try:
        return import_module("openai.resources.responses.responses")
    except ImportError:
        return None
