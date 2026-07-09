OpenTelemetry CrewAI Instrumentation
====================================

This package provides the setup for instrumenting CrewAI with OpenTelemetry
Generative AI semantic conventions. CrewAI operation instrumentation will be
added in follow-up changes.

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

Message content capture can be configured with the
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` environment variable.
