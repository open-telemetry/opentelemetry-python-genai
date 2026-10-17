# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

# pylint: skip-file
import os

from sentence_transformers import SentenceTransformer

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.instrumentation.genai.sentence_transformers import (
    SentenceTransformersInstrumentor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# configure tracing
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter())
)

# configure metrics
metrics.set_meter_provider(
    MeterProvider(
        metric_readers=[
            PeriodicExportingMetricReader(OTLPMetricExporter()),
        ]
    )
)

# instrument sentence-transformers
SentenceTransformersInstrumentor().instrument()


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
