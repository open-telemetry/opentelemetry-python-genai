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

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Alternatively, content capture can be configured programmatically by passing a
``ContentCapturingMode`` as ``content_capturing_mode`` to ``instrument()``.
The supplied mode takes precedence over the environment variable. Omit it or
pass ``None`` to use the environment setting (default: ``NO_CONTENT``):

.. code-block:: python

    from opentelemetry.util.genai.types import ContentCapturingMode

    CrewAIInstrumentor().instrument(
        content_capturing_mode=ContentCapturingMode.SPAN_AND_EVENT,
    )
