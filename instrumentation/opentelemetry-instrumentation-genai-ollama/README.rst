OpenTelemetry Ollama Instrumentation
====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-ollama.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-ollama/

This library allows tracing chat and embedding requests made with the
`Ollama Python library <https://github.com/ollama/ollama-python>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-ollama

Usage
-----

.. code:: python

    import ollama

    from opentelemetry.instrumentation.genai.ollama import OllamaInstrumentor

    OllamaInstrumentor().instrument()

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

Instrumented methods
--------------------

* ``Client.chat`` and ``AsyncClient.chat``, including ``stream=True``:
  ``chat`` spans with request options, token usage, and finish reasons.
* ``Client.embed`` and ``AsyncClient.embed``: ``embeddings`` spans with the
  embedding dimension count and input token usage.
* The module-level ``ollama.chat`` and ``ollama.embed`` functions.

A ``chat`` call without messages only loads or unloads the model, and an
``embed`` call without input embeds nothing, so neither is recorded.

``gen_ai.provider.name`` is ``ollama``, and ``server.address`` and
``server.port`` come from the client's host.

Known limitations
-----------------

* ``generate`` and the deprecated ``embeddings`` methods are not instrumented.
* A function imported with ``from ollama import chat`` before ``instrument()``
  is called is bound to the uninstrumented client. Call ``instrument()``
  first, or use ``ollama.chat``.
* For tools passed as Python functions, only the tool name is recorded.
* Tool call ids are not recorded: the Ollama Python library does not expose
  them on responses or send them in requests.
* Images attached to messages are not captured.

Configuration
-------------

Message content capture is controlled by
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``. Supported values are
``no_content``, ``span_only``, ``event_only``, and ``span_and_event``.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `GenAI Semantic Conventions <https://github.com/open-telemetry/semantic-conventions-genai>`_
