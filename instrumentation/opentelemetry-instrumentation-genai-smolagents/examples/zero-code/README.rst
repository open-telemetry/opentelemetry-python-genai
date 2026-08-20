OpenTelemetry smolagents Zero-Code Instrumentation Example
==========================================================

This is an example of how to instrument smolagents agent runs with zero code
changes, using `opentelemetry-instrument`.

When `main.py <main.py>`_ is run, it exports traces, metrics, and logs to an
OTLP compatible endpoint. The agent run produces an ``invoke_agent`` span, with
one ``execute_tool`` span nested under it per tool call.

Note: `.env <.env>`_ file configures additional environment variables:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=span_only`` configures the smolagents instrumentation to capture prompt and completion contents on *span* attributes.
- ``OTEL_LOGS_EXPORTER=otlp`` to specify exporter type.
- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` (commented out) - uncomment along with ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`` to upload prompts and completions to an ``fsspec``-compatible destination instead of recording them inline. Also uncomment the ``opentelemetry-util-genai[upload]`` line in `requirements.txt <requirements.txt>`_ and reinstall.

Setup
-----

Minimally, update the `.env <.env>`_ file with your ``HF_TOKEN``. An OTLP
compatible endpoint should be listening for traces, metrics, and logs on
http://localhost:4317. If not, update ``OTEL_EXPORTER_OTLP_ENDPOINT`` as well.

Next, set up a virtual environment like this:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install "python-dotenv[cli]"
    pip install -r requirements.txt

Run
---

Run the example like this:

::

    dotenv run -- opentelemetry-instrument python main.py

You should see the agent's answer while traces, metrics, and logs export to
your configured observability tool.
