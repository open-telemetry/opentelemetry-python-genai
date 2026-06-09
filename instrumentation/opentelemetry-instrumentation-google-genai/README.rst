OpenTelemetry Google GenAI SDK Instrumentation
==============================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-google-genai.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-google-genai/

This library adds instrumentation to the `Google GenAI SDK library <https://pypi.org/project/google-genai/>`_
to emit telemetry data following `Semantic Conventions for GenAI systems <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_.
It adds trace spans for GenAI operations, events/logs for recording prompts/responses, and emits metrics that describe the
GenAI operations in aggregate.


Experimental
------------

This package is still experimental. The instrumentation may not be complete or correct just yet.

Please see "TODOS.md" for a list of known defects/TODOs that are blockers to package stability.


Installation
------------

If your application is already instrumented with OpenTelemetry, add this
package to your requirements.
::

    pip install opentelemetry-instrumentation-google-genai

If you don't have a Google GenAI SDK application, yet, try our `examples <examples>`_.

Check out `zero-code example <examples/zero-code>`_ for a quick start.


Usage
-----

This section describes how to set up Google GenAI SDK instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.


Instrumenting all clients
*************************

When using the instrumentor, all clients will automatically trace GenAI ``generate_content`` operations.
You can also optionally capture prompts and responses as log events.

Make sure to configure OpenTelemetry tracing, logging, metrics, and events to capture all telemetry emitted by the instrumentation.

.. code-block:: python

    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
    from google.genai import Client

    GoogleGenAiSdkInstrumentor().instrument()

    client = Client()
    response = client.models.generate_content(
        model="gemini-1.5-flash-002",
        contents="Write a short poem on OpenTelemetry."
    )


Limitations
***********

When using the Google GenAI SDK with automatic function calling enabled,
the OpenTelemetry instrumentation creates a span only for the top-level
``generate_content`` call.

Internal model or tool calls triggered automatically by the SDK are executed
within the SDK internals and are not traced as separate spans.


Enabling message content
************************

Message content is not captured by default. To capture message content set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of ``NO_CONTENT``, ``SPAN_AND_EVENT``, ``SPAN_ONLY``, ``EVENT_ONLY``.
This controls whether the following content is captured on spans and/or events:

- Input messages to the LLM.
- Output messages from the LLM.
- System Instructions
- The result of tool calls and the tool call parameters (other tool call details and tool definitions are always captured).


Configuration recording
***********************

The instrumentation can optionally record ``GenerateContentConfig`` parameters
as span and event attributes under the ``gcp.gen_ai.operation.config.*`` namespace.

By default, no config fields are recorded. You can control which fields are
captured using the following environment variables:

* ``OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_INCLUDES`` — A comma-separated
  list of config field names to include in the span attributes. For example:

  .. code-block:: bash

      export OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_INCLUDES=temperature,max_output_tokens

* ``OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_EXCLUDES`` — A comma-separated
  list of config field names to exclude from the span attributes:

  .. code-block:: bash

      export OTEL_GOOGLE_GENAI_GENERATE_CONTENT_CONFIG_EXCLUDES=stop_sequences

If both variables are set, the includes list is applied first, then the
excludes list filters the result further.

Uninstrument
************

To uninstrument clients, call the uninstrument method:

.. code-block:: python

    from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

    GoogleGenAiSdkInstrumentor().instrument()
    # ...

    # Uninstrument all clients
    GoogleGenAiSdkInstrumentor().uninstrument()


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Google Gen AI SDK (Python) <https://github.com/googleapis/python-genai>`_
* `Google Gen AI SDK Documentation <https://ai.google.dev/gemini-api/docs/sdks>`_
* `Using Vertex AI with Google Gen AI SDK <https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview>`_
