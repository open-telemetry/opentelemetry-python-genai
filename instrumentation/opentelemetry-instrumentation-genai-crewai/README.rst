OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI with OpenTelemetry Generative AI semantic
conventions. It emits:

* ``invoke_agent`` spans for ``Agent.execute_task``, ``Agent.aexecute_task``,
  ``Agent.kickoff``, and ``Agent.kickoff_async`` calls.
* ``execute_tool`` spans for direct/native ``BaseTool.run`` calls and
  structured/ReAct ``CrewStructuredTool.invoke`` calls.

Model calls are not instrumented at the CrewAI layer because CrewAI delegates
them to LiteLLM or provider SDKs. Enable the corresponding client
instrumentation to emit ``chat`` spans without double-counting model calls.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-crewai

Usage
-----

::

    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()

Until Crew and Flow workflow spans are added, sequential agent invocations are
separate root traces unless the application supplies a parent span. For
example:

::

    from opentelemetry import trace

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("my crew run"):
        crew.kickoff()

Configuration
-------------

By default, prompts and completions are not captured. To capture message
content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to ``SPAN_ONLY`` or
``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY

Agent and tool invocations record content as span attributes only; they emit
no event logs, so ``EVENT_ONLY`` captures nothing for this instrumentation.

Captured content can also be forwarded to external storage with a completion
hook. Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` together with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``, or provide a hook directly:

::

    CrewAIInstrumentor().instrument(completion_hook=my_completion_hook)

The programmatic argument takes precedence over the environment variable.

CrewAI's built-in telemetry
---------------------------

CrewAI's built-in telemetry is left unchanged and can run alongside this
instrumentation. To opt out, use CrewAI's own configuration before importing
CrewAI (or this package, which imports it):

::

    export CREWAI_DISABLE_TELEMETRY=true

CrewAI releases before 1.15 install their telemetry ``TracerProvider`` as the
global provider when ``crewai`` is imported. On those releases, applications
that configure OpenTelemetry through the global provider must set
``CREWAI_DISABLE_TELEMETRY=true`` before the import; otherwise a later
``set_tracer_provider()`` call is ignored. Explicit providers passed to
``CrewAIInstrumentor().instrument(...)`` continue to work without that
environment variable.

CrewAI's opt-in cloud tracing (``CREWAI_TRACING_ENABLED`` or
``Crew(tracing=True)``) is also left unchanged.

Current limitations
-------------------

Crew and Flow ``invoke_workflow`` spans, tool call IDs on the ReAct path (where
CrewAI exposes no ID), memory retrieval, and Flow node spans are not yet
instrumented. Native function-calling tool IDs are recorded.

Recursive ``Agent.execute_task`` retries emit one nested ``invoke_agent`` span
per method invocation; retries initiated by a task guardrail outside that
method emit sibling spans.
