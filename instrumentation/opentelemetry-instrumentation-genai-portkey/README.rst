OpenTelemetry Portkey AI Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-portkey.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-portkey/

This library provides OpenTelemetry instrumentation for `Portkey AI <https://github.com/Portkey-AI/portkey-python-sdk>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-portkey

Usage
-----

.. code-block:: python

    from portkey_ai import Portkey
    from opentelemetry.instrumentation.genai.portkey import PortkeyInstrumentor

    # Instrument Portkey AI
    PortkeyInstrumentor().instrument()

    client = Portkey(api_key="PORTKEY_API_KEY")

    # Chat completion example
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    # Prompt completion example
    prompt_response = client.prompts.completions.create(
        prompt_id="YOUR_PROMPT_ID",
        variables={"user_input": "Hello!"},
    )

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``no_content``, ``span_only``, ``event_only``, or ``span_and_event``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_and_event

Uploading Prompts and Completions
*********************************

To enable the built-in upload completion hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible URI/path
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``).

Install the ``upload`` extra to pull in ``fsspec``::

    pip install opentelemetry-util-genai[upload]

You can also programmatically pass a custom hook when calling ``instrument()``:

.. code-block:: python

    PortkeyInstrumentor().instrument(completion_hook=my_custom_hook)

Uninstrument
************

To remove instrumentation from Portkey AI clients, call ``uninstrument()``:

.. code-block:: python

    PortkeyInstrumentor().uninstrument()

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Portkey Documentation <https://portkey.ai/docs>`_
* `Portkey Python SDK Repository <https://github.com/Portkey-AI/portkey-python-sdk>`_
