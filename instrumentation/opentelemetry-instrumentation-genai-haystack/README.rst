OpenTelemetry Haystack Instrumentation
=======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-haystack.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-haystack/

This library allows tracing GenAI operations performed with the
`Haystack <https://haystack.deepset.ai/>`_ Python framework: LLM generator calls and embedder calls.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-haystack

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.haystack import HaystackInstrumentor

    # Instrument Haystack
    HaystackInstrumentor().instrument()


What gets instrumented
***********************

- Components classified as a generator, embedder, or embedder -- one span per component ``run`` / ``run_async`` call.
  Classification is a best-effort read of the component's class name and
  ``run`` method type hints, since Haystack has no static component-kind
  marker. Components that don't fall into one of these (prompt builders,
  routers, converters, ...) are not wrapped: ``opentelemetry-util-genai``
  has no invocation type for a generic pipeline step. Component classes are
  classified the instant they're registered (hooking the ``@component``
  decorator itself), so instrumenting before importing your pipeline's
  components works correctly.

See ``tests/conformance/`` for the exact operations covered.

Known limitations
*****************

- ``gen_ai.response.id`` is not populated for real OpenAI-backed chat
  generators: Haystack's own ``OpenAIChatGenerator`` does not copy the
  provider response id into the reply's ``meta``, so it's only populated for
  generators (or tests) that do include it there.
- ``server.address`` / ``server.port`` are only populated once a component's
  underlying SDK client has been constructed. ``Pipeline.run()`` calls
  ``warm_up()`` automatically, so this is available for pipeline-driven
  calls; a component called standalone only gets it starting on its second
  call, since nothing else triggers ``warm_up()`` first.
- ``gen_ai.provider.name`` has no mapping for Hugging Face API
  generators/embedders (their model string encodes the provider, but there's
  no corresponding enum value yet); it's set to ``"unknown"`` for these.
- Only components classified as a generator, embedder, or embedder are wrapped -- there's no ``opentelemetry-util-genai`` invocation
  type for a generic pipeline step (prompt builders, routers, converters,
  joiners, ...).
- Per-document embedded text/vectors aren't recorded -- ``EmbeddingInvocation``
  only carries aggregate request/response metadata.

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT


Uploading prompts and completions
***********************************

Instead of recording message content inline, prompts and completions can be uploaded to external
storage via a completion hook. To enable the built-in upload hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible URI/path
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``), and install the ``upload`` extra
  (``pip install opentelemetry-util-genai[upload]``).

A custom ``CompletionHook`` can also be passed programmatically, taking precedence over the
environment variable::

    HaystackInstrumentor().instrument(completion_hook=my_hook)


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Haystack <https://haystack.deepset.ai/>`_
* `Haystack documentation <https://docs.haystack.deepset.ai/>`_
