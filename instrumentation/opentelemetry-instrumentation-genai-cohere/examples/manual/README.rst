OpenTelemetry Cohere Instrumentation Example
============================================

This is a placeholder for an example showing how to instrument Cohere
calls when configuring OpenTelemetry SDK and Instrumentations manually.

The ``CohereInstrumentor`` shipped in this release is scaffold-only:
calling ``.instrument()`` sets up tracer, logger, and meter providers
but does not yet wrap any Cohere client methods. Chat completions and
streaming wrapping land in a follow-up PR.

Setup
-----

Minimally, update the `.env <.env>`_ file with your ``CO_API_KEY``. An
OTLP compatible endpoint should be listening for traces and logs on
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

    dotenv run -- python main.py

Once the follow-up PR lands, you will see Cohere chat completions
exported as spans and logs to your configured observability tool.
