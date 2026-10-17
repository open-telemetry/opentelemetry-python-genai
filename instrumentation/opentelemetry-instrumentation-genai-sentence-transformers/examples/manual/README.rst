OpenTelemetry Sentence Transformers Instrumentation Example
===========================================================

This is an example of how to instrument Sentence Transformers when configuring
OpenTelemetry SDK and Instrumentations manually.

When `main.py <main.py>`_ is run, it exports traces and metrics to an OTLP
compatible endpoint. Traces include details such as the model used, the
embedding dimension count, and the duration of the ``encode`` call. The model
runs locally, so no API key is required.

Note: `.env <.env>`_ file configures additional environment variables:

- ``OTEL_SERVICE_NAME`` names the service the telemetry is reported under.
- ``EMBEDDING_MODEL`` (commented out) selects a different model.

Setup
-----

An OTLP compatible endpoint should be listening for traces and metrics on
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

The first run downloads the ``all-MiniLM-L6-v2`` model (~90 MB). You should
see the embedding shape printed while traces and metrics export to your
configured observability tool.
