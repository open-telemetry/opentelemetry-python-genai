OpenTelemetry Amazon Bedrock Instrumentation
============================================

This package provides OpenTelemetry instrumentation for Amazon Bedrock (via the AWS SDK for Python, ``boto3``),
implementing the OpenTelemetry Generative AI semantic conventions.

Supported Operations
--------------------

* Synchronous chat via the Converse API (``client.converse``)
* Streaming chat via the ConverseStream API (``client.converse_stream``)
* Model invocation via the InvokeModel API (``client.invoke_model``)
* Streaming model invocation via the InvokeModelWithResponseStream API (``client.invoke_model_with_response_stream``)

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-bedrock

Usage
-----

.. code-block:: python

    import boto3
    from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor

    # Enable instrumentation
    BedrockInstrumentor().instrument()

    # Use Bedrock runtime client normally
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="amazon.nova-micro-v1:0",
        messages=[{"role": "user", "content": [{"text": "Hello, Bedrock!"}]}],
    )

Configuration
-------------

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

Prompts and completions can also be redirected via a completion hook by
setting ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` or by passing
``instrument(completion_hook=...)``.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Amazon Bedrock Documentation <https://docs.aws.amazon.com/bedrock/>`_
* `Boto3 Documentation <https://boto3.amazonaws.com/v1/documentation/api/latest/index.html>`_
