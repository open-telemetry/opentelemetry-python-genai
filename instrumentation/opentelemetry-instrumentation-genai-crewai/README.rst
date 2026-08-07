OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI LLM calls using the OpenTelemetry Generative AI
semantic conventions. It emits inference spans and client duration and token
usage metrics from CrewAI's public LLM lifecycle events.

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

CrewAI's native telemetry is disabled while this instrumentation is active to
avoid emitting two independent sets of spans. To retain CrewAI's native
telemetry, explicitly enable it before instrumenting::

    export CREWAI_DISABLE_TELEMETRY=false

An existing ``CREWAI_DISABLE_TELEMETRY`` value is always preserved. When the
instrumentation supplies the default value, it removes that value again during
``uninstrument()``.

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Completion hooks
----------------

To forward captured prompts and completions to the built-in upload hook, set
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` and configure an
``fsspec``-compatible destination with
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``::

    export OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload
    export OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=/path/to/prompts

Install ``opentelemetry-util-genai[upload]`` to use the upload hook. A hook can
also be supplied programmatically; it takes precedence over the environment
variable::

    CrewAIInstrumentor().instrument(completion_hook=my_hook)

See ``examples/custom_hook.py`` for a minimal custom hook.
