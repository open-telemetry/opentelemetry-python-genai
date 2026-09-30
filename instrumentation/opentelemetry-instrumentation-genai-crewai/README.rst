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

CrewAI ships anonymous usage telemetry that exports spans to
``telemetry.crewai.com``. It is disabled while this instrumentation is active
and restored on ``uninstrument()``. To keep it running alongside
OpenTelemetry:

::

    CrewAIInstrumentor().instrument(disable_crewai_telemetry=False)

CrewAI releases before 1.15 also install that telemetry's ``TracerProvider`` as
the global provider when ``crewai`` is imported, which ``instrument()`` cannot
undo. On those releases set ``CREWAI_DISABLE_TELEMETRY=true`` before importing
CrewAI or this package (which imports it), otherwise a later
``set_tracer_provider()`` call is ignored.

CrewAI's opt-in cloud tracing (``CREWAI_TRACING_ENABLED`` or
``Crew(tracing=True)``) is not affected by this setting.

Current limitations
-------------------

Crew and Flow ``invoke_workflow`` spans, the async ``Agent.kickoff_async`` and
``Agent.aexecute_task`` APIs, tool call IDs, memory retrieval, and Flow node
spans are not yet instrumented. (``Agent.kickoff`` called while an event loop
is running returns a coroutine; that coroutine is instrumented when awaited.)

Recursive ``Agent.execute_task`` retries emit one nested ``invoke_agent`` span
per method invocation; retries initiated by a task guardrail outside that
method emit sibling spans.
