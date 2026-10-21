Suppressing Inference Events via Declarative Configuration
==========================================================

This example demonstrates how to suppress ``gen_ai.client.inference.operation.details``
log events across GenAI instrumentations using OpenTelemetry's declarative
configuration file format (version ``0.4``).

Overview
--------

Using the ``logger_configurator/development`` component in the OpenTelemetry
declarative configuration file (`configuration.yaml <configuration.yaml>`_),
you can disable loggers matching the wildcard pattern
``opentelemetry.instrumentation*genai*``. This disables logging for all GenAI
instrumentations in this repository, suppressing the inference event (which is the only event instrumentation writes) while
still allowing traces and spans to be collected and exported.

Configuration File
------------------

The configuration is specified in `configuration.yaml <configuration.yaml>`_:

.. code-block:: yaml

    file_format: "0.4"
    logger_provider:
      processors:
        - batch:
            exporter:
              otlp:
                protocol: http/protobuf
      logger_configurator/development:
        loggers:
          # matches all instrumentations in our repo:
          # opentelemetry-instrumentation-genai-* and opentelemetry-instrumentation-google-genai
          - name: "opentelemetry.instrumentation*genai*"
            config:
              enabled: false

Setup
-----

Update `.env <.env>`_ with your ``OPENAI_API_KEY`` (and ensure an OTLP-compatible
collector is listening on http://localhost:4318, or adjust
``OTEL_EXPORTER_OTLP_ENDPOINT``).

Run
---

Run the application using ``uv`` and ``opentelemetry-instrument`` with the configuration file:

.. code-block:: sh

    uv run --env-file .env opentelemetry-instrument --config configuration.yaml python main.py

Alternatively, you can set the config file path using the environment variable:

.. code-block:: sh

    export OTEL_EXPERIMENTAL_CONFIG_FILE=configuration.yaml
    uv run --env-file .env opentelemetry-instrument python main.py

You will see the chat completion result printed in the console. Spans will be
exported as usual, but no ``gen_ai.client.inference.operation.details`` log events
will be emitted.
