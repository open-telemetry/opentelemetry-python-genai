# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the Sentence Transformers instrumentation tests.

``sentence-transformers`` runs the model in the current process, so there is
no HTTP traffic to record. Instead the tests build a real
``SentenceTransformer`` from a ``StaticEmbedding`` module over a small
in-memory tokenizer: ``encode`` runs the library's real code path without
downloading any weights.
"""

from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

try:
    # 6.0+ location; the old one still works there but warns.
    from sentence_transformers.sentence_transformer.modules import (
        StaticEmbedding,
    )
except ImportError:
    from sentence_transformers.models import StaticEmbedding

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_VOCAB = {"[UNK]": 0, "hello": 1, "world": 2, "embedding": 3, "spans": 4}


def sentence_transformer(
    model_name: str | None = MODEL_NAME,
    dimension: int = 8,
    error: Exception | None = None,
) -> SentenceTransformer:
    """A real ``SentenceTransformer`` with in-memory weights.

    ``model_name`` is recorded on ``model_card_data.base_model``, where the
    library stores it for a model loaded from the Hub. ``error`` makes the
    forward pass raise, as a runtime failure inside ``encode`` would.
    """
    tokenizer = Tokenizer(WordLevel(_VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    embedding = StaticEmbedding(tokenizer, embedding_dim=dimension)
    if error is not None:

        def _raise(*_: Any, **__: Any) -> Any:
            raise error

        embedding.forward = _raise

    model = SentenceTransformer(modules=[embedding])
    model.model_card_data.base_model = model_name
    return model
