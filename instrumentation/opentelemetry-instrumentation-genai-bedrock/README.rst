OpenTelemetry Amazon Bedrock Instrumentation
============================================

This package provides the setup for instrumenting Amazon Bedrock with OpenTelemetry
Generative AI semantic conventions. Amazon Bedrock operation instrumentation will be
added in follow-up changes.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-bedrock

Usage
-----

::

    from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor

    BedrockInstrumentor().instrument()

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
