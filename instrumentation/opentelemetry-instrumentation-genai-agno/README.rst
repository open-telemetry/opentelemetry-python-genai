OpenTelemetry Agno Instrumentation
==================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-agno.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-agno/

This library provides OpenTelemetry instrumentation for `Agno <https://github.com/agno-agi/agno>`_.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-agno

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.agno import AgnoInstrumentor

    # Instrument Agno
    AgnoInstrumentor().instrument()

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message content, set the
environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of
``NO_CONTENT``, ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `Agno Documentation <https://docs.agno.com/>`_
* `Agno GitHub Repository <https://github.com/agno-agi/agno>`_
