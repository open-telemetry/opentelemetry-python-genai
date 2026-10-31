OpenTelemetry Util for GenAI
============================

The GenAI Utils package provides boilerplate and helpers to standardize instrumentation for Generative AI.
It offers APIs to minimize the work needed to instrument GenAI libraries,
while providing standardization for generating spans, metrics, and events.


Key Components
--------------

- ``TelemetryHandler`` -- manages LLM invocation lifecycles (spans, metrics, events)
- ``InferenceInvocation`` and message types (``TextPart``, ``ReasoningPart``, ``BlobPart``, etc.) -- structured data model for GenAI interactions
- ``CompletionHook`` -- protocol for uploading content to external storage (built-in ``fsspec`` support)
- Metrics -- ``gen_ai.client.operation.duration`` and ``gen_ai.client.token.usage`` histograms, plus
  the streaming timing histograms ``gen_ai.client.operation.time_to_first_chunk`` and
  ``gen_ai.client.operation.time_per_output_chunk``, recorded for streamed inference calls


Usage
-----

See the module docstring in ``opentelemetry.util.genai.handler`` for usage examples,
including context manager and manual lifecycle patterns.


Context Management and Propagation
----------------------------------

Invocation factory methods on ``TelemetryHandler`` (such as ``inference``, ``workflow``, ``tool``,
``embedding``, ``retrieval``, and ``invoke_local_agent``) accept an optional ``context`` keyword argument
to manage context:

- ``context``: An explicit OpenTelemetry ``Context`` to parent the span. When omitted, the current
  ambient context is used.


Nested Invocation Suppression and Context Enrichment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When an operation runs inside an active operation of the same type, nested spans, metrics,
and events are suppressed to prevent duplicate telemetry. Agent, workflow, and tool invocations are
exempt because those are usually legitimately nested.

When using ``TelemetryHandler`` directly, you get this behavior for free—the outer invocation
places its typed dataclass onto the context, inner invocations suppress telemetry and publish
their attributes into it, and the outer invocation enriches its span and metrics on finish
(outer values take precedence).

Alternatively, you can manage suppression and enrichment directly using OpenTelemetry's
context API with the exported context key and dataclass.

**Outer invocation:**

.. code-block:: python

    from opentelemetry import context
    from opentelemetry.util.genai.invocation import (
        CLIENT_INFERENCE_CONTEXT_KEY,
        InferenceData,
    )

    data = InferenceData()
    token = context.attach(context.set_value(CLIENT_INFERENCE_CONTEXT_KEY, data))
    try:
        # Inner instrumented call detects data on context, suppresses its span,
        # and populates attributes onto data
        execute_operation()
    finally:
        context.detach(token)

    # Attributes captured by inner calls are available on data
    print(data.response_model, data.usage_input_tokens)

**Inner invocation:**

.. code-block:: python

    from opentelemetry import context
    from opentelemetry.util.genai.invocation import (
        CLIENT_INFERENCE_CONTEXT_KEY,
        InferenceData,
    )

    data = context.get_value(CLIENT_INFERENCE_CONTEXT_KEY)
    if isinstance(data, InferenceData):
        # Outer operation is active: suppress span/metrics and publish into shared dataclass
        data.response_model = "gpt-4o-mini"
        data.usage_input_tokens = 10
    else:
        # Emit telemetry normally
        ...

Exported from ``opentelemetry.util.genai.invocation``:

- Inference: ``CLIENT_INFERENCE_CONTEXT_KEY``, ``InferenceData``
- Embedding: ``CLIENT_EMBEDDING_CONTEXT_KEY``, ``EmbeddingData``
- Fetch Response: ``CLIENT_FETCH_RESPONSE_CONTEXT_KEY``, ``FetchResponseData``
- Retrieval: ``CLIENT_RETRIEVAL_CONTEXT_KEY``, ``RetrievalData``

**Content capture and completion upload controls:**

Context data classes accept two boolean parameters (defaulting to ``False``):

- ``capture_content``: When set to ``True`` on the context data, signals that the outer instrumentation permits inner suppressed invocations to capture content attributes (such as input and output messages, system instructions, and tool definitions) and publish them into context.
- ``upload_completion``: When set to ``True`` on the context data, signals that the outer instrumentation permits inner suppressed invocations to trigger completion hook uploads.

.. note::

   Setting these parameters only indicates that the outer instrumentation is OK with the inner instrumentation adding content capture attributes or uploading completions. Whether the inner instrumentation actually does so still depends on the user's settings.




Stream execution context
------------------------

``SyncStreamWrapper`` and ``AsyncStreamWrapper`` scope each stream read and
cleanup operation with ``_execution_context()``. Context is restored before
returning a chunk to the consumer. Generator ``send``/``throw`` and
``asend``/``athrow`` also use this scope when the underlying stream supports them.

For invocation-backed streams, pass the invocation to the base constructor
(``super().__init__(stream, invocation)``) and call ``invocation.suspend()``
before returning the stream. The base wrapper then makes the suspended invocation
current during reads, cleanup and finalization of an abandoned stream, so no
``_execution_context()`` override is needed.
Keep finalization in the existing ``_on_stream_end`` and ``_on_stream_error`` hooks.


Modalities
----------

``opentelemetry.util.genai.types.Modality`` provides string enum members
``TEXT``, ``IMAGE``, ``VIDEO``, ``AUDIO``, and ``DOCUMENT``. Use these constants
when constructing ``BlobPart``, ``FilePart``, and ``UriPart`` or passing
modality/token-count pairs to the inference invocation's token setters:

.. code-block:: python

    from opentelemetry.util.genai.types import Modality, UriPart

    image = UriPart(
        mime_type="image/png",
        modality=Modality.IMAGE,
        uri="https://example.com/image.png",
    )

Members serialize and format as their string values, such as ``"image"``.
Message parts continue to accept plain strings, including provider-specific
modalities. Token setters continue to record only ``text``, ``image``, and
``audio``; other modalities are ignored.

``Modality`` replaces the previous ``Literal`` type alias. Annotations that
also accept raw strings should use ``Modality | str`` rather than ``Modality``
alone. Custom modalities remain strings, not additional enum members.


Retrieval Documents
-------------------

Set ``RetrievalInvocation.documents`` using
``opentelemetry.util.genai.types.RetrievalDocument`` objects:

.. code-block:: python

    from opentelemetry.util.genai.types import RetrievalDocument

    with handler.retrieval(data_source_id="my-index") as invocation:
        invocation.documents = [RetrievalDocument(id="doc-1", score=0.9)]

The model contains only the optional ``id`` and ``score`` fields; unset
fields serialize as JSON ``null``. Documents are recorded in
``gen_ai.retrieval.documents`` only in ``SPAN_ONLY`` or ``SPAN_AND_EVENT``
content-capture mode. Passing dictionaries is deprecated, but existing
dictionary payloads continue to serialize unchanged.


Environment Variables
---------------------

This package relies on environment variables to configure capturing of message content.
By default, message content will not be captured.
Set the environment variable ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` to one of:

- ``NO_CONTENT``: Do not capture message content (default).
- ``SPAN_ONLY``: Capture message content in spans only.
- ``EVENT_ONLY``: Capture message content in events only.
- ``SPAN_AND_EVENT``: Capture message content in both spans and events.

When a ``LoggerProvider`` is configured, the package emits ``gen_ai.client.inference.operation.details``
events (with ``DEBUG`` severity) regardless of the content capturing mode. Message content is
included (as dictionaries instead of JSON strings) only in ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` mode.
To suppress these events while keeping a ``LoggerProvider`` active, configure your logger or
``LogRecordProcessor`` to filter them out (see `examples/suppress-inference-event
<https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/examples/suppress-inference-event>`_).

When a client operation (inference, embeddings, retrieval, fetch response, or remote agent
invocation) fails with an exception, the package also emits a ``gen_ai.client.operation.exception``
event (with ``WARN`` severity) carrying ``exception.type``, ``exception.message``, and
``exception.stacktrace``. ``exception.message`` may contain sensitive information. The event is not
emitted for failures reported without an exception, for abandoned streams, or for a
``BaseException`` that is not an ``Exception``, such as cancellation.

Completion Hook / Upload
^^^^^^^^^^^^^^^^^^^^^^^^

- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``: Name of the completion hook entry point to load (e.g. ``upload``).
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH``: An ``fsspec``-compatible URI/path for uploading prompts and completions
  (e.g. ``/path/to/prompts`` or ``gs://my_bucket``). Required when using the ``upload`` hook.
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_FORMAT``: Format for uploaded data -- ``json`` (default) or ``jsonl``.
- ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_MAX_QUEUE_SIZE``: Maximum number of concurrent uploads to queue (default: ``20``).


Span Attributes
---------------

This package sets the following span attributes on LLM invocations:

**Common attributes:**

- ``gen_ai.operation.name``: Str(chat)
- ``gen_ai.provider.name``: Str(openai)
- ``gen_ai.request.model``: Str(gpt-4o)
- ``server.address``: Str(api.openai.com)
- ``server.port``: Int(443)

**Response attributes:**

- ``gen_ai.response.finish_reasons``: Slice(["stop"])
- ``gen_ai.response.model``: Str(gpt-4o-2024-05-13)
- ``gen_ai.response.id``: Str(chatcmpl-Bz8yrvPnydD9pObv625n2CGBPHS13)
- ``gen_ai.usage.input_tokens``: Int(24)
- ``gen_ai.usage.output_tokens``: Int(7)
- ``gen_ai.usage.cache_write.input_tokens``: Int(10)
- ``gen_ai.usage.cache_read.input_tokens``: Int(5)

**Request parameter attributes (when provided):**

- ``gen_ai.request.temperature``: Float(0.7)
- ``gen_ai.request.top_p``: Float(1.0)
- ``gen_ai.request.frequency_penalty``: Float(0.0)
- ``gen_ai.request.presence_penalty``: Float(0.0)
- ``gen_ai.request.max_tokens``: Int(1024)
- ``gen_ai.request.stop_sequences``: Slice(["\\n"])
- ``gen_ai.request.seed``: Int(42)

**Content attributes (sensitive, requires content capturing enabled):**

- ``gen_ai.input.messages``: Str('[{"role": "user", "parts": [{"content": "hello world", "type": "text"}]}]')
- ``gen_ai.output.messages``: Str('[{"role": "assistant", "parts": [{"content": "hello back", "type": "text"}], "finish_reason": "stop"}]')
- ``gen_ai.system_instructions``: Str('[{"content": "You are a helpful assistant.", "type": "text"}]')

**Error attributes:**

- ``error.type``: Str(TimeoutError)

Embedding Span Attributes
-------------------------

This package also supports embedding invocation spans via the ``embedding`` context manager.
For embedding invocations, the following attributes are set:

**Common attributes:**

- ``gen_ai.operation.name``: Str(embeddings)
- ``gen_ai.provider.name``: Str(openai)
- ``server.address``: Str(api.openai.com)
- ``server.port``: Int(443)

**Request attributes:**

- ``gen_ai.request.model``: Str(text-embedding-3-small)
- ``gen_ai.embeddings.dimension.count``: Int(1536)
- ``gen_ai.request.encoding_formats``: Slice(["float"])

**Response attributes:**

- ``gen_ai.response.model``: Str(text-embedding-3-small)
- ``gen_ai.usage.input_tokens``: Int(24)


Capturing request iterators
---------------------------

Use ``IteratorCapture.from_property()`` for a named request property. It replaces
that property's value in ``kwargs`` and records values as the SDK reads them.
Disabled captures and missing properties leave ``kwargs`` untouched and return
``None`` from ``snapshot()``.

.. code-block:: python

    from opentelemetry.util.genai.iterator_capture import IteratorCapture

    messages_capture = IteratorCapture.from_property(
        kwargs, "messages", enabled=capture_content
    )
    try:
        response = sdk_call(**kwargs)
    finally:
        messages = messages_capture.extract(get_input_messages)

For a bare value, pass ``capture.wrapped`` to the SDK:

.. code-block:: python

    capture = IteratorCapture(messages, enabled=capture_content)
    try:
        response = sdk_call(messages=capture.wrapped)
    finally:
        messages = capture.extract(get_input_messages)

``extract(extractor)`` calls the instrumentation's extractor with a snapshot,
using ``()`` for unread iterators and ``None`` for disabled or missing properties.
The extractor must accept that shape and return the telemetry content.
``extract()`` logs snapshot or conversion errors and returns ``None``. Use
``snapshot()`` directly when no conversion is needed.

Nested built-in dictionaries, lists, and tuples are supported. File objects
pass through unchanged. Snapshots contain only the consumed portion of each
iterator; async iterators and custom containers are not captured.


Installation
------------

::

    pip install opentelemetry-util-genai

For upload support (requires ``fsspec``)::

    pip install opentelemetry-util-genai[upload]


Design Document
---------------

The design document for the OpenTelemetry GenAI Utils can be found at: `Design Document <https://docs.google.com/document/d/1LzNGylxot5zaIV1goOJZ2mz3LI0weFu1SgaMnyDv7gg/edit?usp=sharing>`_

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry GenAI semantic conventions <https://opentelemetry.io/docs/specs/semconv/gen-ai/>`_
