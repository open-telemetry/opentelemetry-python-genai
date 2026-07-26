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
app``), or — if you control an embedding process — instrument
programmatically:

.. code-block:: python

    from opentelemetry.instrumentation.genai.qwenpaw import QwenPawInstrumentor

    QwenPawInstrumentor().instrument()
    # ... run the QwenPaw app in this process ...
    QwenPawInstrumentor().uninstrument()

Configuration
-------------

Capture Message Content
***********************

By default, input and output messages are not captured. To enable message
content capture, set the environment variable:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true

Conformance
-----------

Semantic-convention conformance scenarios live in `tests/conformance
<tests/conformance>`_.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `QwenPaw <https://github.com/agentscope-ai/QwenPaw>`_
