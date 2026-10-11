# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval execution for DSPy."""

from __future__ import annotations

from typing import Any

import numpy as np
from dspy.retrievers.embeddings import EmbeddingsWithScores


def fake_embedder(texts: list[str]) -> Any:
    vecs = []
    for t in texts:
        if "OpenTelemetry" in t:
            vecs.append([1.0, 0.0])
        else:
            vecs.append([0.0, 1.0])
    return np.array(vecs, dtype=np.float32)


corpus = [
    "OpenTelemetry is an observability framework.",
    "DSPy is a declarative framework for language models.",
]
emb_retriever = EmbeddingsWithScores(
    corpus=corpus, embedder=fake_embedder, k=1, brute_force_threshold=10
)
emb_retriever("What is OpenTelemetry?")
