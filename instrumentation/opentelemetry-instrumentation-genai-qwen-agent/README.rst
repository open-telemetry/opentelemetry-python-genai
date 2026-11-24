OpenTelemetry Qwen-Agent Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-qwen-agent.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-qwen-agent/

This library allows tracing agent runs and tool executions made through the
`Qwen-Agent framework <https://pypi.org/project/qwen-agent/>`_.

It produces spans following the `GenAI semantic conventions
<https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_:

- ``invoke_agent`` for ``Agent.run()``
- ``execute_tool`` for ``Agent._call_tool()``

Each ``invoke_agent`` span also reports how many model and tool calls the run
made, through the ``gen_ai.invoke_agent.inference_calls`` and
``gen_ai.invoke_agent.tool_calls`` histograms. Model calls are counted per
``Agent._call_llm()`` invocation, so retries performed inside the model client
are not double counted, and tool calls count executed tools (including
``Memory`` file retrieval and failed executions, excluding unknown tools
rejected before execution).

LLM call (``chat``) spans are intentionally not emitted by this
instrumentation: the model client libraries qwen-agent calls into have their
own instrumentations, and emitting them here as well would duplicate the LLM
spans. Enable the instrumentation of the underlying model client library
(e.g. `opentelemetry-instrumentation-genai-openai
<https://pypi.org/project/opentelemetry-instrumentation-genai-openai/>`_ for
OpenAI-compatible backends) alongside this package to capture LLM calls.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-qwen-agent

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.genai.qwen_agent import (
        QwenAgentInstrumentor,
    )
    from qwen_agent.agents import Assistant

    QwenAgentInstrumentor().instrument()

    bot = Assistant(
        llm={"model": "qwen-max", "model_type": "qwen_dashscope"},
        name="my-assistant",
    )
    for responses in bot.run([{"role": "user", "content": "Hello!"}]):
        pass

Message content capture is disabled by default. Set
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT`` to record
prompts, completions, and tool arguments/results.

Alternatively, content capture can be configured programmatically by passing a
``ContentCapturingMode`` as ``content_capturing_mode`` to ``instrument()``.
The supplied mode takes precedence over the environment variable. Omit it or
pass ``None`` to use the environment setting (default: ``NO_CONTENT``):

.. code-block:: python

    from opentelemetry.util.genai.types import ContentCapturingMode

    QwenAgentInstrumentor().instrument(
        content_capturing_mode=ContentCapturingMode.SPAN_AND_EVENT,
    )

Uploading prompts and completions
---------------------------------

Instead of recording message content inline, prompts and completions can be
uploaded to external storage via a completion hook. To enable the built-in
upload hook, set:

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload``
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to an ``fsspec``-compatible
  URI/path (e.g. ``/path/to/prompts`` or ``gs://my_bucket``), and install the
  ``upload`` extra (``pip install opentelemetry-util-genai[upload]``).

A custom ``CompletionHook`` can also be passed programmatically, taking
precedence over the environment variable::

    QwenAgentInstrumentor().instrument(completion_hook=my_hook)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Qwen-Agent <https://github.com/QwenLM/Qwen-Agent>`_
