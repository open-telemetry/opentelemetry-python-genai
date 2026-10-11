# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


class _ConformanceRetriever(BaseRetriever):
    def __init__(self) -> None:
        super().__init__()
        self.similarity_top_k = 1

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        return [
            NodeWithScore(
                node=TextNode(
                    id_="capital-france",
                    text="Paris is the capital of France.",
                ),
                score=0.99,
            )
        ]


_ConformanceRetriever().retrieve("What is the capital of France?")
