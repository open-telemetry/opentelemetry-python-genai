OpenTelemetry Anthropic Instrumentation
=======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-anthropic.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-anthropic/

This library allows tracing LLM requests made by the
`Anthropic Python SDK <https://pypi.org/project/anthropic/>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-anthropic

If you don't have an Anthropic application yet, try our `examples <examples>`_
which only need a valid Anthropic API key.

Check out the `zero-code example <examples/zero-code>`_ for a quick start.

Usage
-----

This section describes how to set up Anthropic instrumentation if you're setting OpenTelemetry up manually.
Check out the `manual example <examples/manual>`_ for more details.

.. code-block:: python

    from opentelemetry.instrumentation.genai.anthropic import AnthropicInstrumentor
    import anthropic

    # Instrument Anthropic
    AnthropicInstrumentor().instrument()

    # Use Anthropic client as normal
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Hello, Claude!"}
        ]
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
* `Anthropic SDK (Python) <https://github.com/anthropics/anthropic-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_

