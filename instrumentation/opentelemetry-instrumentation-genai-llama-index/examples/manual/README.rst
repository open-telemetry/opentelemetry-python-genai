OpenTelemetry LlamaIndex Instrumentation Example
================================================

This example configures the OpenTelemetry SDK and LlamaIndex instrumentation
manually. It exports traces, logs, and metrics to an OTLP-compatible endpoint.
The trace includes LlamaIndex ``invoke_agent`` and ``execute_tool`` spans plus
the OpenAI inference spans emitted by the provider instrumentation.

Setup
-----

Run an OTLP-compatible collector on ``http://localhost:4317``, or set
``OTEL_EXPORTER_OTLP_ENDPOINT`` to another endpoint. Then create an environment
and install the example dependencies:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export OPENAI_API_KEY=your_api_key_here

Run
---

::

    python agent.py

Set ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` to capture
prompt and completion content on spans.

Completion hooks
----------------

Set ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` and
``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to use the upload hook.
`custom_hook.py <custom_hook.py>`_ demonstrates passing a custom hook directly:

::

    python custom_hook.py
