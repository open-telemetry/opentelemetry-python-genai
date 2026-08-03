OpenTelemetry LlamaIndex Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-llama-index.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-llama-index/

This package contains the OpenTelemetry instrumentor scaffold for
`LlamaIndex <https://github.com/run-llama/llama_index>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-llama-index

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.llama_index import (
        LlamaIndexInstrumentor,
    )

    LlamaIndexInstrumentor().instrument()

Configuration
-------------

Content capture is controlled through the
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment variable.
Supported values are ``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, and
``SPAN_AND_EVENT``.

Prompts and completions can also be redirected via a completion hook by
setting ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` or by passing
``instrument(completion_hook=...)``.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `LlamaIndex Documentation <https://docs.llamaindex.ai/>`_
* `LlamaIndex GitHub Repository <https://github.com/run-llama/llama_index>`_
