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
  ``gen_ai.client.operation.time_per_output_chunk``


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

To control event emission, you can optionally set ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` to ``true`` or ``false`` (case-insensitive).
This variable controls whether to emit ``gen_ai.client.inference.operation.details`` events.
If not explicitly set, the default value is automatically determined by ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT``:

- When ``NO_CONTENT`` or ``SPAN_ONLY`` is set: defaults to ``false``
- When ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` is set: defaults to ``true``

If explicitly set, the user's value takes precedence over the default.

When ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` mode is enabled and a LoggerProvider is configured,
the package also emits ``gen_ai.client.inference.operation.details`` events with structured
message content (as dictionaries instead of JSON strings). Note that when using ``EVENT_ONLY``
or ``SPAN_AND_EVENT``, the ``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT`` environment variable defaults
to ``true``, so events will be emitted automatically unless explicitly set to ``false``.

Native log enablement
^^^^^^^^^^^^^^^^^^^^^

On SDKs that provide ``Logger.enabled``, inference event export also honors native
log enablement. The check uses the invocation's context and event name. Disabling
export does not disable spans, metrics, or completion hooks. Hooks still receive
the event record when the content-capture configuration requests one.

For example, pass your log exporter to this processor to suppress inference events:

.. code-block:: python

    from opentelemetry._logs import SeverityNumber
    from opentelemetry.context import Context
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    class DropInferenceEvents(BatchLogRecordProcessor):
        def enabled(
            self,
            *,
            context: Context | None = None,
            instrumentation_scope: InstrumentationScope | None = None,
            severity_number: SeverityNumber | None = None,
            event_name: str | None = None,
        ) -> bool:
            return (
                event_name != "gen_ai.client.inference.operation.details"
                and super().enabled(
                    context=context,
                    instrumentation_scope=instrumentation_scope,
                    severity_number=severity_number,
                    event_name=event_name,
                )
            )

    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(DropInferenceEvents(exporter))

Pass this provider as ``logger_provider`` when configuring instrumentation. If the
provider has several processors, each must disable the event: the SDK considers it
enabled when any processor accepts it.

``OTEL_INSTRUMENTATION_GENAI_EMIT_EVENT=false`` still disables these events.
SDKs without ``Logger.enabled`` retain the existing environment-variable behavior.

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