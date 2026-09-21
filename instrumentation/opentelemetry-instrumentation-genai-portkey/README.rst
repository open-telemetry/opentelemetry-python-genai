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

Token Usage
***********

Chat and prompt completions record detailed usage when Portkey returns it,
for synchronous, asynchronous, and streaming calls:

- ``usage.prompt_tokens_details.cache_write_tokens`` records
  ``gen_ai.usage.cache_write.input_tokens``, falling back to
  ``usage.cache_creation_input_tokens`` when the nested count is unavailable.
- ``usage.prompt_tokens_details.cached_tokens`` records
  ``gen_ai.usage.cache_read.input_tokens``, falling back to
  ``usage.cache_read_input_tokens`` when the nested count is unavailable.
  The two representations are not added together.
- Explicit ``text_tokens``, ``image_tokens``, and ``audio_tokens`` under
  ``usage.prompt_tokens_details`` record the corresponding
  ``gen_ai.usage.{text,image,audio}.input_tokens`` attributes.
- Explicit ``text_tokens``, ``image_tokens``, and ``audio_tokens`` under
  ``usage.completion_tokens_details`` record
  ``gen_ai.usage.{text,image,audio}.output_tokens``.
- ``usage.completion_tokens_details.reasoning_tokens`` records
  ``gen_ai.usage.reasoning.output_tokens`` without adding to the reported
  output-token total.

Availability depends on the provider and model. Aggregate input and output
totals are preserved, and modality counts are never inferred from those
totals. No per-modality cache counts are inferred from the aggregate cache-read
count. Only non-negative integer counts, excluding booleans, are accepted.
Missing or invalid counts are omitted; streaming updates
retain previously reported counts when later chunks omit them. These
attributes do not require message-content capture.

The fallback cache fields follow Portkey's
`prompt caching response format <https://portkey.ai/docs/integrations/llms/anthropic/prompt-caching>`_;
nested details include fields preserved by its OpenAI-compatible response
passthrough.

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
