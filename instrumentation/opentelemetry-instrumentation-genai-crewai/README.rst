OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI-owned orchestration operations using the
OpenTelemetry Generative AI semantic conventions. It emits ``invoke_workflow``
spans for crew kickoffs and ``invoke_agent`` spans for agent executions.

CrewAI delegates model calls to LLM client libraries. This instrumentation does
not emit inference spans for those calls; instrument the underlying LLM client
to capture model-call telemetry without duplicate spans.

The instrumentation adds its own handlers to CrewAI's event bus. It does not
disable CrewAI telemetry, change CrewAI environment variables, or remove event
handlers registered by the application or other integrations.

To emit only OpenTelemetry GenAI telemetry, users can disable CrewAI's native
telemetry before importing CrewAI while leaving this instrumentation enabled::

    import os

    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

    from crewai import Crew
    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()

Do not use ``OTEL_SDK_DISABLED`` for this purpose because it disables the
OpenTelemetry SDK globally, including the GenAI telemetry emitted by this
instrumentation.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-crewai

Usage
-----

::

    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()

Configuration
-------------

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT
