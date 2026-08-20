OpenTelemetry Groq Instrumentation
==================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-groq.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-groq/

This library allows tracing LLM requests and logging of messages made by the
`Groq Python API library <https://pypi.org/project/groq/>`_. It also captures
the duration of the operations and the number of tokens used as metrics.

Installation
------------

If your application is already instrumented with OpenTelemetry, add this
package to your requirements.
::

    pip install opentelemetry-instrumentation-genai-groq

If you don't have a Groq application, yet, try our `examples <examples>`_
which only need a valid Groq API key.

Check out `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Groq instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

Instrumenting all clients
*************************

When using the instrumentor, all clients will automatically trace Groq operations including chat completions.
You can also optionally capture prompts and completions as log events.

Make sure to configure OpenTelemetry tracing, logging, and events to capture all telemetry emitted by the instrumentation.

.. code-block:: python

    from opentelemetry.instrumentation.genai.groq import GroqInstrumentor
    from groq import Groq

    GroqInstrumentor().instrument()

    client = Groq()
    # Chat completion example
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )

Enabling message content
*************************

Message content such as the contents of the prompt, completion, function arguments and return values
are not captured by default. To capture message content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of the following values:

- ``span_only`` - capture content on *span* attributes.
- ``event_only`` - capture content on *event* attributes.
- ``span_and_event`` - capture content on both *span* and *event* attributes.
- ``no_content`` - do not capture content (the default).

Uploading prompts and completions
*********************************

To enable the built-in upload hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible URI/path
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``).

Install the ``upload`` extra to pull in ``fsspec``::

    pip install opentelemetry-util-genai[upload]

See the `opentelemetry-util-genai
<https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/util/opentelemetry-util-genai/README.rst>`_
for additional options.

Enabling the latest experimental features
***********************************************

The latest experimental GenAI semantic conventions are used unconditionally; there is
no environment variable to opt in or out.

.. note:: Generative AI semantic conventions are still evolving. The latest experimental features may introduce breaking changes in future releases.

Uninstrument
************

To uninstrument clients, call the uninstrument method:

.. code-block:: python

    from opentelemetry.instrumentation.genai.groq import GroqInstrumentor

    GroqInstrumentor().instrument()
    # ...

    # Uninstrument all clients
    GroqInstrumentor().uninstrument()

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Groq SDK (Python) <https://github.com/groq/groq-python>`_
