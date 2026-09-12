OpenTelemetry Cohere Instrumentation
====================================

This library allows tracing LLM requests made by the
`Cohere Python SDK <https://github.com/cohere-ai/cohere-python>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-cohere

If you don't have a Cohere application yet, try our `examples <examples>`_
which only need a valid Cohere API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Cohere instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.genai.cohere import CohereInstrumentor
    import cohere

    # Instrument Cohere
    CohereInstrumentor().instrument()

    # Use the Cohere V2 client as normal
    co = cohere.ClientV2()
    response = co.chat(
        model="command-r-plus-08-2024",
        messages=[{"role": "user", "content": "hello world!"}],
    )
    print(response)


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
* `Cohere Python SDK <https://github.com/cohere-ai/cohere-python>`_
* `Cohere Documentation <https://docs.cohere.com/>`_
