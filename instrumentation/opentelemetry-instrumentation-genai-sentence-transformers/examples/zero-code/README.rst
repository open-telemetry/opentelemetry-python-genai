OpenTelemetry Sentence Transformers Zero-Code Instrumentation Example
=====================================================================

This is an example of how to instrument Sentence Transformers with zero code
changes, using ``opentelemetry-instrument``.

When `main.py <main.py>`_ is run, it exports traces and metrics to an OTLP
compatible endpoint. Traces include details such as the model used, the
embedding dimension count, and the duration of the ``encode`` call. The model
runs locally, so no API key is required.

Note: `.env <.env>`_ file configures additional environment variables:

- ``OTEL_SERVICE_NAME`` names the service the telemetry is reported under.
- ``OTEL_TRACES_EXPORTER=console`` and ``OTEL_METRICS_EXPORTER=console``
  (commented out) print telemetry to the terminal instead of exporting it,
  which needs no OTLP endpoint.

Setup
-----

An OTLP compatible endpoint should be listening for traces and metrics on
http://localhost:4317. If not, update ``OTEL_EXPORTER_OTLP_ENDPOINT`` or
uncomment the console exporters in `.env <.env>`_.

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

The first run downloads the ``all-MiniLM-L6-v2`` model (~90 MB). You should
see the embedding shape printed while traces and metrics export to your
configured observability tool.
