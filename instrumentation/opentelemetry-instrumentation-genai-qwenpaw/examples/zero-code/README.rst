OpenTelemetry QwenPaw Zero-Code Instrumentation Example
=======================================================

This is an example of how to use OpenTelemetry's automatic instrumentation
(zero-code) capabilities with QwenPaw.

QwenPaw is started as its own app through the ``qwenpaw`` CLI, so there is
no ``main.py`` here — the ``opentelemetry-instrument`` CLI wraps the app's
own entrypoint and instruments it without any code changes. Every user turn
handled by the agent runner exports an ``invoke_agent`` span, along with
logs and metrics, to an OTLP compatible endpoint.

Note: `.env <.env>`_ file configures additional environment variables:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY`` configures
  QwenPaw instrumentation to capture the turn's input and output messages on
  span attributes.

Setup
-----

An OTLP compatible endpoint should be listening for traces, logs, and metrics
on http://localhost:4317. If not, update "OTEL_EXPORTER_OTLP_ENDPOINT" as
well.

Next, set up a virtual environment like this:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install "python-dotenv[cli]"
    pip install -r requirements.txt

You will also need a model provider API key (DashScope by default). Set it
as an environment variable:

::

    export DASHSCOPE_API_KEY=your_api_key_here

Then initialize the QwenPaw working directory (config and agent defaults)
once:

::

    qwenpaw init

Run
---

Run the QwenPaw app with zero-code instrumentation like this:

::

    dotenv run -- opentelemetry-instrument qwenpaw app

Open the QwenPaw console (http://127.0.0.1:8088 by default) and send a
message. Traces, logs, and metrics export to your configured observability
tool — no changes to QwenPaw were required!

Learn More
----------

See the `OpenTelemetry Python automatic instrumentation docs
<https://opentelemetry.io/docs/languages/python/automatic/>`_ for more
information about zero-code instrumentation.
