Suppressing Inference Events in OpenTelemetry GenAI
===================================================

1. `Declarative Configuration <declarative/>`_:
   Using OpenTelemetry's declarative configuration file (`configuration.yaml <declarative/configuration.yaml>`_),
   disable loggers matching the pattern ``"opentelemetry.instrumentation*genai*"`` via
   ``logger_configurator/development``.
2. `Programmatic (Manual) Filtering <manual/>`_:
   Using Python code (`main.py <manual/main.py>`_), wrap your
   ``LogRecordProcessor`` with a custom processor that intercepts and drops
   the ``gen_ai.client.inference.operation.details`` event.

Examples
--------

- `declarative/ <declarative/>`_: Zero-code configuration with ``opentelemetry-instrument`` and YAML.
- `manual/ <manual/>`_: In-code SDK configuration using a custom ``LogRecordProcessor``.
