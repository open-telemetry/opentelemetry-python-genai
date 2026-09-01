OpenTelemetry LlamaIndex Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-llama-index.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-llama-index/

This package contains OpenTelemetry instrumentation for
`LlamaIndex <https://github.com/run-llama/llama_index>`_.

It emits ``invoke_workflow`` spans for ``AgentWorkflow`` runs,
``invoke_agent`` spans for standalone and workflow-member ``FunctionAgent``
and ``ReActAgent`` executions, and ``execute_tool`` spans when LlamaIndex
executes tools. It also emits ``retrieval`` spans for synchronous and
asynchronous ``BaseRetriever`` operations. Model and embedding calls
delegated to provider SDKs are intentionally left to those SDKs' OpenTelemetry
instrumentations.

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
setting ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` together with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``, or by passing a custom hook
directly with ``instrument(completion_hook=...)``. See
`examples/manual/custom_hook.py <examples/manual/custom_hook.py>`_ for a
programmatic example.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `LlamaIndex Documentation <https://docs.llamaindex.ai/>`_
* `LlamaIndex GitHub Repository <https://github.com/run-llama/llama_index>`_
