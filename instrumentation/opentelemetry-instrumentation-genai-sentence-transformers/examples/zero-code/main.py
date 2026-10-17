# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import os

from sentence_transformers import SentenceTransformer


def main():
    # Runs locally; the model is downloaded once and then cached.
    model = SentenceTransformer(
        os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    sentences = [
        "OpenTelemetry provides observability for your applications.",
        "Sentence Transformers computes text embeddings locally.",
    ]
    embeddings = model.encode(sentences)

    print(f"Model: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding sample (first 5 dimensions): {embeddings[0][:5]}")


if __name__ == "__main__":
    main()
