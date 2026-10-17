# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
Sentence Transformers instrumentation supporting the `sentence-transformers
<https://pypi.org/project/sentence-transformers/>`_ library, it can be enabled
by using ``SentenceTransformersInstrumentor``.

``sentence-transformers`` runs the embedding model in-process (there is no
remote API call), so this instrumentation records an ``embeddings`` span for
each ``SentenceTransformer.encode`` call with ``gen_ai.provider.name`` set to
``huggingface``. It is distinct from
``opentelemetry-instrumentation-transformers`` (Traceloop), which traces text
*generation* with the base Hugging Face ``transformers`` library.

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

    SentenceTransformersInstrumentor().uninstrument()

API
---
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.genai.sentence_transformers.package import (
    _instruments,
)
from opentelemetry.instrumentation.genai.sentence_transformers.patch import (
    wrap_encode,
)
from opentelemetry.instrumentation.genai.sentence_transformers.version import (
    __version__,
)
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

__all__ = ["SentenceTransformersInstrumentor", "__version__"]

_MODULE = "sentence_transformers"
_CLASS = "SentenceTransformer"

# Only ``encode`` is wrapped. ``encode_query`` and ``encode_document`` (added
# in 5.0) are prompt-specialized convenience methods that delegate to
# ``encode`` internally, so wrapping them as well would emit a second,
# redundant span for the same embedding call. Wrapping ``encode`` alone yields
# exactly one span per user-facing call on every supported version.
_ENCODE_METHODS = ("encode",)


def _with_handler(
    wrapper: Callable[..., Any], handler: TelemetryHandler
) -> Callable[..., Any]:
    def _wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        return wrapper(wrapped, instance, args, kwargs, handler)

    return _wrapper


class SentenceTransformersInstrumentor(BaseInstrumentor):
    """OpenTelemetry instrumentor for the ``sentence-transformers`` library.

    Instruments ``SentenceTransformer.encode``, emitting one ``embeddings``
    span per call. The ``encode_query`` / ``encode_document`` convenience
    methods delegate to ``encode``, so they are covered without being wrapped
    separately (which would double-count).
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable Sentence Transformers instrumentation."""
        handler = TelemetryHandler(
            tracer_provider=kwargs.get("tracer_provider"),
            meter_provider=kwargs.get("meter_provider"),
            logger_provider=kwargs.get("logger_provider"),
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
            instrumentation_scope_name=__package__,
            instrumentation_scope_version=__version__,
        )

        import sentence_transformers

        cls = getattr(sentence_transformers, _CLASS)
        wrapper = _with_handler(wrap_encode, handler)
        for method in _ENCODE_METHODS:
            if hasattr(cls, method):
                # Positional arguments for wrapt 1.x/2.x compatibility (wrapt
                # 2.x renamed the first parameter from ``module`` to
                # ``target``).
                wrap_function_wrapper(
                    _MODULE,
                    f"{_CLASS}.{method}",
                    wrapper,
                )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Disable Sentence Transformers instrumentation."""
        import sentence_transformers

        cls = getattr(sentence_transformers, _CLASS)
        for method in _ENCODE_METHODS:
            if hasattr(cls, method):
                unwrap(cls, method)
