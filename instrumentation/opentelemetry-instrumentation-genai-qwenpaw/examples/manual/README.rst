OpenTelemetry QwenPaw Instrumentation Example
=============================================

This is an example of how to instrument QwenPaw when configuring the
OpenTelemetry SDK and instrumentations manually in an embedding process.

When `main.py <main.py>`_ is run, it starts the QwenPaw app in-process and
exports traces, logs, and metrics to an OTLP compatible endpoint. Every user
turn handled by the agent runner produces an ``invoke_agent`` span carrying
the agent id, agent name, and session id. Pair it with the AgentScope
instrumentation to also capture the model and tool calls QwenPaw delegates.

`custom_hook.py <custom_hook.py>`_ shows the same setup with a custom
``CompletionHook`` that receives the captured prompts and completions, as an
alternative to recording them inline.

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

Run the example like this:

::

    dotenv run -- python main.py

Open the QwenPaw console (http://127.0.0.1:8088 by default) and send a
message. Each turn exports an ``invoke_agent`` span, along with logs and
metrics, to your configured observability tool.

Run the custom completion hook variant like this:

::

    dotenv run -- python custom_hook.py
