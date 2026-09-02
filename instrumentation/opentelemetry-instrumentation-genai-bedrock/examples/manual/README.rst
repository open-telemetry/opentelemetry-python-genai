OpenTelemetry Amazon Bedrock Instrumentation Example
====================================================

This is an example of how to instrument Amazon Bedrock calls when configuring OpenTelemetry SDK and Instrumentations manually.

When `main.py <main.py>`_ is run, it exports traces and logs to an OTLP
compatible endpoint.

Note: `.env <.env>`_ file configures additional environment variables:

- `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY` configures
  Amazon Bedrock instrumentation to capture prompt and completion contents on
  span attributes.

Setup
-----

An OTLP compatible endpoint should be listening for traces and logs on
http://localhost:4317. If not, update "OTEL_EXPORTER_OTLP_ENDPOINT" as well.

Next, set up a virtual environment like this:

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install "python-dotenv[cli]"
    pip install -r requirements.txt

You will also need AWS credentials configured (e.g. via environment variables or an AWS profile):

::

    export AWS_DEFAULT_REGION=us-east-1
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key

Run
---

Run the example like this:

::

    python main.py
