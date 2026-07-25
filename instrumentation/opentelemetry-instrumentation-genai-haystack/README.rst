OpenTelemetry Haystack Instrumentation
=======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-haystack.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-haystack/

This library allows tracing GenAI operations performed with the
`Haystack <https://haystack.deepset.ai/>`_ Python framework: pipeline
execution, LLM generator calls, embedder calls, and retriever/ranker calls.

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

    # Use Haystack as normal
    from haystack import Pipeline
    from haystack.components.generators.chat.openai import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    pipeline = Pipeline()
    pipeline.add_component("llm", OpenAIChatGenerator(model="gpt-4o"))
    pipeline.run({"llm": {"messages": [ChatMessage.from_user("Hello!")]}})

What gets instrumented
***********************

- ``Pipeline.run`` / ``Pipeline.run_async`` -- one ``invoke_workflow`` span per
  pipeline execution.
- Components classified as a generator, embedder, or retriever/ranker --
  one span per component ``run`` / ``run_async`` call. Classification is a
  best-effort read of the component's class name and ``run`` method type
  hints, since Haystack has no static component-kind marker. Components
  that don't fall into one of these (prompt builders, routers, converters,
  agents, tool invokers, ...) are not wrapped: ``opentelemetry-util-genai``
  has no invocation type for a generic pipeline step. See
  ``tests/conformance/`` for the exact operations covered and this
  package's ``MIGRATION_REPORT.md`` for the full gap list.

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
