OpenTelemetry AWS Bedrock Instrumentation
==========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-bedrock.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-bedrock/

This library allows tracing LLM requests made via the
`AWS Bedrock Runtime <https://docs.aws.amazon.com/bedrock/latest/APIReference/>`_
``Converse`` and ``ConverseStream`` APIs using ``botocore``.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-bedrock

Usage
-----

This section describes how to set up AWS Bedrock instrumentation if you're setting OpenTelemetry up manually.

.. code-block:: python

    from opentelemetry.instrumentation.genai.bedrock import BedrockInstrumentor
    import boto3

    # Instrument Bedrock
    BedrockInstrumentor().instrument()

    # Use Bedrock Runtime client as normal
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.converse(
        modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=[
            {"role": "user", "content": [{"text": "Hello, Claude!"}]}
        ],
    )


Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To enable message content capture,
set the environment variable:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `AWS Bedrock Documentation <https://docs.aws.amazon.com/bedrock/>`_
* `botocore <https://github.com/boto/botocore>`_
