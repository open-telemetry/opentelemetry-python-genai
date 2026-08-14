OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI-owned orchestration operations using the
OpenTelemetry Generative AI semantic conventions. It emits ``invoke_workflow``
spans for crew kickoffs and ``invoke_agent`` spans for agent executions.

CrewAI delegates model calls to LLM client libraries. This instrumentation does
not emit inference spans for those calls; instrument the underlying LLM client
to capture model-call telemetry without duplicate spans.

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
