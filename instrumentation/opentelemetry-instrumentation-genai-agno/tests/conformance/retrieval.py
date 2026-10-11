# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval via Knowledge.search for Agno."""

from __future__ import annotations

from unittest.mock import MagicMock

from agno.knowledge.knowledge import Document, Knowledge

kb = Knowledge(name="conformance-kb", max_results=2)
kb.vector_db = MagicMock()
kb.vector_db.name = "conformance_vector_db"
kb.vector_db.provider = "pgvector"
kb.vector_db.search.return_value = [
    Document(
        content="OpenTelemetry provides observability standards.",
        id="doc-1",
        meta_data={"source": "docs"},
        reranking_score=0.95,
    ),
    Document(
        content="Agno provides multi-agent framework.",
        id="doc-2",
        meta_data={"source": "docs"},
        reranking_score=0.88,
    ),
]
kb.search(query="what is OpenTelemetry?", max_results=2)
