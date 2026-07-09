OpenTelemetry CrewAI Instrumentation
====================================

This package instruments CrewAI with OpenTelemetry GenAI semantic conventions.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-crewai

Usage
-----

::

    from opentelemetry.instrumentation.genai.crewai import CrewAIInstrumentor

    CrewAIInstrumentor().instrument()

The instrumentation creates GenAI workflow, agent, tool, and retrieval spans for
CrewAI crews, flows, agents, tasks, tools, and memory operations.
