OpenTelemetry QwenPaw Instrumentation
=====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-qwenpaw.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-qwenpaw/

This library traces user turns handled by `QwenPaw
<https://github.com/agentscope-ai/QwenPaw>`_, a personal assistant
application built on AgentScope. Each turn that goes through
``AgentRunner.query_handler`` (1.x) or ``Runtime.run`` (2.x) produces one ``invoke_agent`` span following
the OpenTelemetry GenAI semantic conventions, carrying the agent display
name (``gen_ai.agent.name``, when already cached by the runner), the session id
(``gen_ai.conversation.id``), and — when content capture is enabled — the
turn's input and output messages.

The request's session ID is propagated to nested GenAI invocations through
OpenTelemetry context.

The span remains open until the response stream is drained or closed.
Successful finalization records ``gen_ai.response.finish_reasons`` when the
last emitted assistant message is complete. Closing a partial message
preserves captured output without reporting a finish reason. Local agent
turns record the ``gen_ai.invoke_agent.duration`` metric.

Abandoning a partially consumed stream ends its span during garbage
collection with ``error.type`` set to ``_OTHER`` and the status description
``abandoned stream``. Captured partial output is preserved without reporting
a finish reason. This requires ``opentelemetry-util-genai`` 1.3b0 or newer
(``1.3b0.dev`` from this workspace during development).

QwenPaw delegates model (LLM) and tool execution to AgentScope, so this
package emits no ``chat`` or ``execute_tool`` spans and its conformance
suite covers none — those operations belong to the AgentScope
instrumentation. Pair this package with the instrumentations of AgentScope
and the underlying model libraries so their calls appear as child spans
under the ``invoke_agent`` span.

Supported versions
------------------

QwenPaw >= 1.1.0, < 3.0.0. Version 1.x uses the agent runner; version
2.x uses the runtime request entry point and captures text, reasoning,
tool calls/results, and image/audio/video/file content from request messages and
streamed SSE envelopes. Request normalization and span creation remain lazy;
SDK-generated session IDs propagate to nested GenAI invocations.
Version 2.x support requires ``opentelemetry-util-genai`` 1.4b0 or newer
(``1.4b0.dev`` from this workspace during development). QwenPaw 2.x requires Python >= 3.11,
< 3.14. Both entry points produce one internal ``invoke_agent`` span per
turn; model and tool execution remain owned by the underlying libraries.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-qwenpaw

Usage
-----

QwenPaw is started as its own app (CLI / process entrypoint). Use
OpenTelemetry zero-code instrumentation (``opentelemetry-instrument qwenpaw
app``) — check out the `zero-code example <examples/zero-code>`_ for a quick
start — or, if you control an embedding process, instrument
programmatically:

.. code-block:: python

    from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor

    QwenPawInstrumentor().instrument()
    # ... run the QwenPaw app in this process ...
    QwenPawInstrumentor().uninstrument()

Check out the `manual example <examples/manual>`_ for more details.

Configuration
-------------

Capture Message Content
***********************

By default, input and output messages are not captured. To capture message
content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
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

    QwenPawInstrumentor().instrument(completion_hook=my_hook)

See `examples/manual/custom_hook.py <examples/manual/custom_hook.py>`_ for a
runnable custom-hook example.

Conformance
-----------

Semantic-convention conformance scenarios live in `tests/conformance
<tests/conformance>`_.

The same ``QwenPawInstrumentor`` selects the entry point for the installed
QwenPaw major version. The 1.x runner instrumentation remains supported.
Tests cover both the oldest and latest releases of each runtime generation:

.. list-table::
   :header-rows: 1

   * - QwenPaw
     - Entry point
     - Conformance tox environment
   * - 1.1.0
     - ``AgentRunner.query_handler``
     - ``py311-test-instrumentation-genai-qwenpaw-oldest-conformance``
   * - Latest 1.x
     - ``AgentRunner.query_handler``
     - ``py313-test-instrumentation-genai-qwenpaw-v1-conformance``
   * - 2.0.0
     - ``Runtime.run``
     - ``py311-test-instrumentation-genai-qwenpaw-v2-conformance``
   * - Latest 2.x
     - ``Runtime.run``
     - ``py313-test-instrumentation-genai-qwenpaw-conformance``

Each version runs successful and failed turns through Weaver live-check.
The 2.x scenarios additionally cover reasoning, tool calls/results, and
multimodal request messages.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `QwenPaw <https://github.com/agentscope-ai/QwenPaw>`_
