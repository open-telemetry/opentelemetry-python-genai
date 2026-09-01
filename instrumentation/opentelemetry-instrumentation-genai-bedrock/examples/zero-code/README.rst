OpenTelemetry Amazon Bedrock Zero-Code Instrumentation Example
==============================================================

This is an example of how to use OpenTelemetry's automatic instrumentation
(zero-code) capabilities with the Amazon Bedrock SDK (boto3).

The `opentelemetry-instrument` CLI automatically instruments your Python
application without requiring code changes. When `main.py <main.py>`_ is run
with the CLI, it exports traces and logs to an OTLP compatible endpoint.

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

Run the example with zero-code instrumentation like this:

::

    dotenv run -- opentelemetry-instrument python main.py
