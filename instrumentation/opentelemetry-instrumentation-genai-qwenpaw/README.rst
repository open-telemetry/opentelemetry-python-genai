OpenTelemetry QwenPaw Instrumentation
=====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-qwenpaw.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-qwenpaw/

This library traces user turns handled by `QwenPaw
<https://github.com/agentscope-ai/QwenPaw>`_, a personal assistant
application built on AgentScope. Each turn that goes through
``AgentRunner.query_handler`` produces one ``invoke_agent`` span following
the OpenTelemetry GenAI semantic conventions, carrying the agent id
(``gen_ai.agent.id``), the agent display name (``gen_ai.agent.name``, on
versions that expose it), the session id (``gen_ai.conversation.id``), and
— when content capture is enabled — the turn's input and output messages.

QwenPaw delegates model (LLM) and tool execution to AgentScope, so this
package emits no ``chat`` or ``execute_tool`` spans and its conformance
suite covers none — those operations belong to the AgentScope
instrumentation. Pair this package with the instrumentations of AgentScope
and the underlying model libraries so their calls appear as child spans
under the ``invoke_agent`` span.

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

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `QwenPaw <https://github.com/agentscope-ai/QwenPaw>`_
