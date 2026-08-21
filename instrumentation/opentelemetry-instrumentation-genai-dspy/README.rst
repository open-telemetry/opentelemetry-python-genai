OpenTelemetry DSPy Instrumentation
==================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-dspy.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-dspy/

This package provides the setup for instrumenting the
`DSPy framework <https://pypi.org/project/dspy/>`_ with OpenTelemetry
Generative AI semantic conventions. DSPy operation instrumentation
will be added in follow-up changes.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-dspy

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.genai.dspy import (
        DSPyInstrumentor,
    )

    DSPyInstrumentor().instrument()

Message content capture is disabled by default. Set
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT`` to record
prompts, completions, and module inputs/outputs.

Uploading prompts and completions
---------------------------------

Instead of recording message content inline, prompts and completions can be
uploaded to external storage via a completion hook. To enable the built-in
upload hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible
  URI/path (e.g. ``/path/to/prompts`` or ``gs://my_bucket``), and install the
  ``upload`` extra (``pip install opentelemetry-util-genai[upload]``).

A custom ``CompletionHook`` can also be passed programmatically, taking
precedence over the environment variable::

    DSPyInstrumentor().instrument(completion_hook=my_hook)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `DSPy <https://github.com/stanfordnlp/dspy>`_
