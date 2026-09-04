OpenTelemetry Cohere Zero-Code Instrumentation Example
======================================================

This is a placeholder for an example showing how to instrument Cohere
calls with zero code changes, using ``opentelemetry-instrument``.

The ``CohereInstrumentor`` shipped in this release is scaffold-only:
the entry point is registered under ``opentelemetry_instrumentor`` so
``opentelemetry-instrument`` discovers it, but no Cohere client methods
are wrapped yet. Chat completions wrapping lands in a follow-up PR.

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

Run the example with zero-code instrumentation like this:

::

    dotenv run -- opentelemetry-instrument python main.py

Once the follow-up PR lands, you will see Cohere chat completions
exported as spans and logs without any changes to ``main.py``.

Learn More
----------

See the `OpenTelemetry Python automatic instrumentation docs
<https://opentelemetry.io/docs/languages/python/automatic/>`_ for more
information about zero-code instrumentation.
