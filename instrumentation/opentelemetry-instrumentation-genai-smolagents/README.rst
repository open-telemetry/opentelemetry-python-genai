OpenTelemetry smolagents Instrumentation
========================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-smolagents.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-smolagents/

This library provides OpenTelemetry instrumentation for `smolagents
<https://github.com/huggingface/smolagents>`_. It emits GenAI
semantic-convention spans and the matching metrics through
``opentelemetry-util-genai``:

* ``invoke_agent`` for ``MultiStepAgent.run()``, streaming and non-streaming
* ``execute_tool`` for a tool call
* ``chat`` for a call to a model class that runs inference in your own process:
  ``TransformersModel``, ``VLLMModel`` and ``MLXModel``

The API-backed model classes are not instrumented here. Each one calls a client
library that carries its own instrumentation. Emitting a span at the smolagents
layer as well would produce two ``chat`` spans for one model call, and would
count the token-usage and duration metrics twice. Install the instrumentation
for the client library instead:

.. list-table::
   :header-rows: 1

   * - smolagents model class
     - Instrument this instead
   * - ``OpenAIModel``, ``AzureOpenAIModel``
     - `opentelemetry-instrumentation-genai-openai
       <https://pypi.org/project/opentelemetry-instrumentation-genai-openai/>`_
   * - ``AmazonBedrockModel``
     - `opentelemetry-instrumentation-botocore
       <https://pypi.org/project/opentelemetry-instrumentation-botocore/>`_
   * - ``InferenceClientModel``, ``LiteLLMModel``, ``LiteLLMRouterModel``
     - the instrumentation or built-in telemetry of the client library the model
       calls (``huggingface_hub``, ``litellm``)

A ``CodeAgent`` with ``executor_type="local"`` passes generated code to
smolagents' ``local_python_executor.timeout()``, which runs it in a worker
thread. This package wraps ``timeout()`` so that a tool span started in that
thread stays nested under the agent span.

``TransformersModel`` is the only instrumented class with a ``generate_stream``.
A streamed call gets a ``chat`` span that stays open until the caller drains the
deltas. This covers both ``stream_outputs=True`` on an agent and a direct
``generate_stream`` call. The span carries ``gen_ai.request.stream``, and the
call also records the
``gen_ai.client.operation.time_to_first_chunk`` and
``gen_ai.client.operation.time_per_output_chunk`` metrics.

Known gaps:

* A subclass that inherits ``generate`` or ``generate_stream`` from one of the
  three classes above is instrumented. A subclass that overrides one is not: the
  override shadows the patched method, so the call produces no ``chat`` span.
* A ``chat`` span reports no ``gen_ai.response.id``, no
  ``gen_ai.response.model``, no ``gen_ai.response.finish_reasons`` and no
  ``server.address``. A runtime in this process returns the generated text and
  the token counts, nothing more. It also listens on no socket.
* ``InferenceClientModel`` calls ``huggingface_hub``, which has no
  instrumentation to enable. An agent running it reports no model call and no
  token usage.
* There is no span for an individual agent step, so ``execute_tool`` spans nest
  directly under ``invoke_agent``. The GenAI semantic conventions define no
  operation for one iteration of a reason-and-act loop. ``invoke_workflow`` is
  the closest name, but both the OpenAI Agents and LangChain instrumentations
  use it for the outermost orchestration, above ``invoke_agent``. A dedicated
  span is proposed in
  `semantic-conventions-genai#81
  <https://github.com/open-telemetry/semantic-conventions-genai/issues/81>`_.
* The agent's step budget (``max_steps``) is not recorded. The GenAI semantic
  conventions define no attribute for it, and a run that uses the budget up
  already reports ``gen_ai.response.finish_reasons`` as ``length`` on the
  ``invoke_agent`` span.
* ``gen_ai.tool.call.id`` is omitted when a single step calls the *same* tool
  more than once. ``Tool.__call__`` receives only the argument values, so this
  instrumentation matches a call to its id by tool name, and two calls to one
  tool in the same step have no unambiguous match. A ``CodeAgent`` has no tool
  call ids at all: its model writes code rather than tool calls.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-smolagents

Usage
-----

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )
    from smolagents import CodeAgent, TransformersModel

    SmolagentsInstrumentor().instrument()

    model = TransformersModel(model_id="HuggingFaceTB/SmolLM2-135M-Instruct")
    agent = CodeAgent(tools=[], model=model)
    agent.run("How many seconds are in a week?")

Configuration
-------------

Capture Message Content
***********************

By default, prompts and completions are not captured. To capture message
content, set the environment variable
``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of ``NO_CONTENT``,
``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT``:

::

    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_AND_EVENT

Uploading Content to External Storage
*************************************

Captured prompt and completion content can be forwarded to external storage
through a completion hook instead of being recorded inline. Select the built-in
``upload`` hook and point it at a destination:

::

    export OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload
    export OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=/path/to/prompts  # or gs://my_bucket

The ``upload`` hook is provided by ``opentelemetry-util-genai`` and requires its
``[upload]`` extra. See the `opentelemetry-util-genai README
<https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/util/opentelemetry-util-genai/README.rst>`_
for the other content-capture and upload options it owns.

You can also pass a hook programmatically, which takes precedence over the
environment variable:

.. code-block:: python

    from opentelemetry.instrumentation.genai.smolagents import (
        SmolagentsInstrumentor,
    )

    SmolagentsInstrumentor().instrument(completion_hook=my_hook)

Check out the `manual example <examples/manual>`_ for a runnable script that
configures the SDK in code, and its ``custom_hook.py`` for the completion hook.
The `zero-code example <examples/zero-code>`_ does the same through
``opentelemetry-instrument``.

Conformance
-----------

The scenarios that check this package against the GenAI semantic conventions
live under ``tests/conformance/``.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
* `smolagents Documentation <https://huggingface.co/docs/smolagents>`_
* `smolagents GitHub Repository <https://github.com/huggingface/smolagents>`_
