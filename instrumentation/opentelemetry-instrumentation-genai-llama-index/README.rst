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
asynchronous ``BaseRetriever`` operations. Model calls
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

How instrumentation works
--------------------------

The instrumentor registers a LlamaIndex span handler with its dispatcher. The
handler observes LlamaIndex-owned operations and delegates span creation and
completion to ``opentelemetry-util-genai``. Provider model calls are left to
the provider's instrumentation, so they can be composed without duplicate
inference spans.

.. code-block:: mermaid

    flowchart TD
        A[LlamaIndexInstrumentor.instrument] --> B[LlamaIndex dispatcher]
        B --> C{Span callback}
        C -->|AgentWorkflow.run| D[workflow invocation]
        C -->|BaseWorkflowAgent.run / run_agent_step| E[agent invocation]
        C -->|call_tool / FunctionTool.call| F[tool invocation]
        C -->|BaseRetriever.retrieve / aretrieve| G[retrieval invocation]
        D --> H[TelemetryHandler.workflow]
        E --> I[TelemetryHandler.invoke_local_agent]
        F --> J[TelemetryHandler.tool]
        G --> K[TelemetryHandler.retrieval]
        H --> L[Start OTel span]
        I --> L
        J --> L
        K --> L
        L --> M[Dispatcher exit or error callback]
        M --> N[Set attributes and stop/fail invocation]
        P[Provider SDK instrumentation] -. model calls .-> Q[inference spans]

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
