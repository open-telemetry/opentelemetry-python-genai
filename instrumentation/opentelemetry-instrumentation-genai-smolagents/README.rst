OpenTelemetry smolagents Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-smolagents.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-smolagents/

This library provides OpenTelemetry instrumentation for `smolagents
<https://github.com/huggingface/smolagents>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-smolagents

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )

    # Instrument smolagents
    SmolagentsInstrumentor().instrument()

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message
content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of ``NO_CONTENT``,
``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Uploading Content to External Storage
*************************************

Captured prompt and completion content can be forwarded to external storage
through a completion hook instead of being recorded inline. Select the built-in
``upload`` hook and point it at a destination:

::

    export OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload
    export OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=/path/to/prompts  # or gs://my_bucket

The ``upload`` hook is provided by ``opentelemetry-util-genai`` and requires its
``[upload]`` extra. See the `opentelemetry-util-genai README
<https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/util/opentelemetry-util-genai/README.rst>`_
for the other content-capture and upload options it owns.

You can also pass a hook programmatically, which takes precedence over the
environment variable:

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )

    SmolagentsInstrumentor().instrument(completion_hook=my_hook)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `smolagents Documentation <https://huggingface.co/docs/smolagents>`_
* `smolagents GitHub Repository <https://github.com/huggingface/smolagents>`_
