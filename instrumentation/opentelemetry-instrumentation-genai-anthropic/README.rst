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


Supported Operations
--------------------

The instrumentation supports synchronous and asynchronous calls for:

- ``messages.create``, ``messages.stream``, and ``messages.parse``
- ``beta.messages.create``, ``beta.messages.stream``, and ``beta.messages.parse``
- Raw and streaming response helpers (``with_raw_response`` and ``with_streaming_response``)


Configuration
-------------

Anthropic SDK built-in tracing
******************************

``anthropic`` 2.x and newer ships its own
`OpenTelemetry tracing <https://github.com/anthropics/anthropic-sdk-python#opentelemetry>`_,
enabled by default. Using it together with this instrumentation produces duplicate spans for
each call.

On ``anthropic`` 2.x, prefer the SDK's built-in tracing and uninstall this package::

    pip uninstall opentelemetry-instrumentation-genai-anthropic

If the package is installed as a dependency of something else, disable it with
``OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=anthropic`` instead.

The built-in tracing doesn't emit metrics, log events and doesn't support upload hooks.
To keep this instrumentation instead, disable the Anthropic SDK tracing by setting
``ANTHROPIC_OPEN_TELEMETRY=false`` or passing ``open_telemetry=False`` to the client:

.. code-block:: python

    client = Anthropic(open_telemetry=False)

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT


Uploading prompts and completions
*********************************

Instead of recording message content inline, prompts and completions can be uploaded to external
storage via a completion hook. To enable the built-in upload hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible URI/path
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``), and install the ``upload`` extra
  (``pip install opentelemetry-util-genai[upload]``).

A custom ``CompletionHook`` can also be passed programmatically, taking precedence over the
environment variable::

    AnthropicInstrumentor().instrument(completion_hook=my_hook)


References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Anthropic SDK (Python) <https://github.com/anthropics/anthropic-sdk-python>`_
* `Anthropic Documentation <https://docs.anthropic.com/>`_
