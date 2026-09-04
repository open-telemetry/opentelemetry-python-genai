# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _FakeRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content="Paris is the capital of France.",
                id="doc-1",
                metadata={"source": "wiki"},
            ),
            Document(
                page_content="The Eiffel Tower is located in Paris.",
                id="doc-2",
                metadata={"source": "wiki"},
            ),
        ]

    def _get_ls_params(self, **kwargs: Any) -> Any:
        params = super()._get_ls_params(**kwargs)
        params["ls_vector_store_provider"] = "FakeVectorStore"
        return params


retriever = _FakeRetriever()
retriever.invoke("What is the capital of France?")
