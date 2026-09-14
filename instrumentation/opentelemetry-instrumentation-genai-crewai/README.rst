OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI with OpenTelemetry Generative AI semantic
conventions. It emits:

* ``invoke_agent`` spans for synchronous ``Agent.execute_task`` calls and
  standalone ``Agent.kickoff`` calls.
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

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Captured content can also be forwarded to external storage with a completion
hook. Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` together with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``, or provide a hook directly:

::

    CrewAIInstrumentor().instrument(completion_hook=my_completion_hook)

The programmatic argument takes precedence over the environment variable.

Current limitations
-------------------

Crew and Flow ``invoke_workflow`` spans, async agent APIs, Flow-internal
``Agent.kickoff`` calls made while an event loop is running, tool call IDs,
memory retrieval, and Flow node
spans are not yet instrumented. Recursive ``Agent.execute_task`` retries emit
one nested ``invoke_agent`` span per method invocation; retries initiated by a
task guardrail outside that method emit sibling spans.
