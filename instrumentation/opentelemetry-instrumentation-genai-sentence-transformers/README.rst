OpenTelemetry Sentence Transformers Instrumentation
====================================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-genai-sentence-transformers.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-genai-sentence-transformers/

This library allows tracing embedding calls made with the
`sentence-transformers <https://pypi.org/project/sentence-transformers/>`_
library.

``sentence-transformers`` runs the embedding model in-process, so there is no
remote API call. The instrumentation emits one ``embeddings`` span per
``SentenceTransformer.encode`` call. ``encode_query`` and ``encode_document``
delegate to ``encode``, so they are covered too.

This is distinct from ``opentelemetry-instrumentation-transformers``
(Traceloop OpenLLMetry), which traces text *generation* with the base Hugging
Face ``transformers`` library.

Installation
------------

::

    pip install opentelemetry-instrumentation-genai-sentence-transformers

Usage
-----

.. code:: python

    from opentelemetry.instrumentation.genai.sentence_transformers import (
        SentenceTransformersInstrumentor,
    )
    from sentence_transformers import SentenceTransformer

    SentenceTransformersInstrumentor().instrument()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(["Hello world", "Embedding spans"])

Emitted telemetry
-----------------

Each ``encode`` call produces an ``embeddings {model}`` span and the
``gen_ai.client.operation.duration`` metric, carrying:

* ``gen_ai.operation.name`` = ``embeddings``
* ``gen_ai.provider.name`` = ``huggingface``
* ``gen_ai.request.model`` — best-effort model identifier
* ``gen_ai.embeddings.dimension.count`` — embedding dimensionality

Input texts are never recorded, and no token usage is reported because the
model runs locally without returning usage data.

Examples
--------

* `examples/manual <examples/manual>`_ — configure the SDK in code.
* `examples/zero-code <examples/zero-code>`_ — use ``opentelemetry-instrument``.

References
----------

* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `GenAI Semantic Conventions <https://github.com/open-telemetry/semantic-conventions/tree/main/docs/gen-ai>`_
