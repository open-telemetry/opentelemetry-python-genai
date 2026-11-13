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

The ``interactions`` and ``embed_content`` methods are newly instrumented and may contain issues.
Please treat the telemetry produced by these methods as experimental.

The ``interactions`` API currently does not support automatic function calling, so no ``execute_tool`` spans
are generated.

``interactions.get`` emits a ``fetch_response`` span. A fetch performs no inference, so the token
counts carried by the fetched interaction describe its original generation and are not recorded.

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

When using the instrumentor, all clients will automatically trace GenAI ``generate_content``, ``embed_content``, ``interactions.create``, ``interactions.get``, and ``aio.live.connect`` operations.
You can also optionally capture prompts and responses as log events.

Make sure to configure OpenTelemetry tracing, logging, and metrics to capture all telemetry emitted by the instrumentation.

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

When using the Google GenAI SDK with automatic function calling enabled for ``generate_content``,
the OpenTelemetry instrumentation creates an ``execute_tool`` span for each tool call the SDK executes,
these spans are nested under the ``generate_content`` span.

Only tool calls that the SDK executes are traced, if you disable the automatic function calling and instead
do the function calling in your application, our instrumentation cannot trace them.


Enabling message content
************************

Message content is not captured by default. To capture message content set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of ``NO_CONTENT``, ``SPAN_AND_EVENT``, ``SPAN_ONLY``, ``EVENT_ONLY``.
This controls whether the following content is captured on spans and/or events:

- Input messages to the LLM.
- Output messages from the LLM.
- System Instructions
- The result of tool calls and the tool call parameters (other tool call details and tool definitions are always captured).

Truncating binary payloads (e.g. audio / image blobs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inline binary media (such as realtime PCM audio chunks or inline images) is only buffered and recorded when content capture is enabled, and is stored on ``BlobPart.content`` as raw ``bytes``. For Gemini Live (``aio.live.connect``), ``realtime_inference`` spans do not emit a ``gen_ai.client.inference.operation.details`` log event and the ``gen_ai.client.realtime_session.started`` / ``ended`` events carry no message payload, so Live message and audio content is recorded on spans only in ``SPAN_ONLY`` or ``SPAN_AND_EVENT`` mode (or passed to a configured ``CompletionHook``; ``EVENT_ONLY`` without a completion hook does not buffer or record Live message content).

When messages are emitted as structured ``AnyValue`` maps/lists on log events (for operations that emit ``gen_ai.client.inference.operation.details``), ``BlobPart.content`` is passed to the OpenTelemetry SDK as raw ``bytes``. You can cap the size of each binary payload (and string field) on log events without corrupting the surrounding message structure by configuring ``OTEL_LOGRECORD_ATTRIBUTE_VALUE_LENGTH_LIMIT`` (or ``OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT``) or ``LoggerProvider(log_record_limits=LogRecordLimits(max_log_record_attribute_length=...))``. On spans, message attributes are currently serialized as a single JSON string (with ``BlobPart.content`` base64-encoded), so ``OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT`` or ``SpanLimits(max_span_attribute_length=...)`` truncates the serialized JSON string itself and can make it unparsable until structured span attributes (`open-telemetry/opentelemetry-python-genai#1013 <https://github.com/open-telemetry/opentelemetry-python-genai/pull/1013>`_) land.

.. note::

   Truncating ``bytes`` and ``bytearray`` attribute values requires `open-telemetry/opentelemetry-python#5855 <https://github.com/open-telemetry/opentelemetry-python/pull/5855>`_ (currently in an unreleased version of ``opentelemetry-api``).


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
