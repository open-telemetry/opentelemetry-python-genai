OpenTelemetry OpenAI Agents Instrumentation
===========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-openai-agents.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-openai-agents/

This library traces applications built with the
`OpenAI Agents SDK <https://pypi.org/project/openai-agents/>`_. It registers a
tracing processor with the Agents runtime and turns its trace/span callbacks
into spans that mirror the structure of your agent run:

* **Workflow spans** for an agent run — one per ``Runner.run`` / ``run_sync``
  invocation, named after the run's ``RunConfig.workflow_name``.
* **Agent spans** for each agent invoked within the workflow, including the
  agent name.
* **Tool spans** for function tool calls, capturing the tool arguments and
  result.

The spans nest to reflect the run, so a single ``Runner.run`` produces a
workflow span with the agent and tool calls it triggered as children.

LLM-level spans (``chat`` / ``embeddings``) are **not** emitted by this package.
They are produced by
`opentelemetry-instrumentation-genai-openai <https://pypi.org/project/opentelemetry-instrumentation-genai-openai/>`_
when it is installed and instrumented alongside this package.

.. note::
   This package continues the project previously published as
   ``opentelemetry-instrumentation-openai-agents-v2``.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-openai-agents

Install ``opentelemetry-instrumentation-genai-openai`` as well to also capture
the underlying LLM calls.

See the `examples <examples>`_ directory for runnable ``manual`` and
``zero-code`` scenarios.

Usage
-----

Call ``OpenAIAgentsInstrumentor().instrument()`` once during startup, then run
your agent as usual. The ``Runner.run_sync(...)`` call below is recorded as a
workflow span with the agent and tool spans nested underneath.

.. code-block:: python

    from agents import Agent, Runner, function_tool

    from opentelemetry.instrumentation.genai.openai_agents import (
        OpenAIAgentsInstrumentor,
    )

    OpenAIAgentsInstrumentor().instrument()


    @function_tool
    def get_weather(city: str) -> str:
        return f"The forecast for {city} is sunny with pleasant temperatures."


    assistant = Agent(
        name="Travel Concierge",
        instructions="You are a concise travel concierge.",
        model="<your-model>",
        tools=[get_weather],
    )

    result = Runner.run_sync(
        assistant, "I'm visiting Barcelona this weekend. How should I pack?"
    )
    print(result.final_output)

Configuration
-------------

By default the instrumentor keeps the Agents SDK's built-in trace exporter
(which uploads traces to OpenAI's hosted tracing backend when ``OPENAI_API_KEY``
is set) active alongside OpenTelemetry emission. Pass
``disable_openai_trace_export=True`` to route traces only through OpenTelemetry:

.. code-block:: python

    OpenAIAgentsInstrumentor().instrument(disable_openai_trace_export=True)

The workflow span name comes from the Agents SDK's own
``Runner.run(..., run_config=RunConfig(workflow_name=...))`` knob. Content
capture is controlled through
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment variable.

Prompts and completions can instead be uploaded to external storage via a
completion hook: set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` (install the ``upload`` extra:
``pip install opentelemetry-util-genai[upload]``), or pass a custom
``CompletionHook`` programmatically, which takes precedence over the
environment variable::

    OpenAIAgentsInstrumentor().instrument(completion_hook=my_hook)

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `OpenAI Agents SDK (Python) <https://github.com/openai/openai-agents-python>`_
