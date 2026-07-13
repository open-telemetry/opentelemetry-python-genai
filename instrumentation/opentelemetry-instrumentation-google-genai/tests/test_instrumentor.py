# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GoogleGenAiSdkInstrumentor."""

from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

try:
    from google.genai._interactions.resources.interactions import (
        AsyncInteractionsResource,
        InteractionsResource,
    )
except ImportError:
    from google.genai._gaos.interactions import (
        AsyncInteractions as AsyncInteractionsResource,
    )
    from google.genai._gaos.interactions import (
        Interactions as InteractionsResource,
    )


def test_co_filename_on_wrapped_functions(
    tracer_provider, logger_provider, meter_provider
):
    # ADK is relying on the __code__ attribute to suppress their instrumentation:
    # https://github.com/google/adk-python/blob/0d4d3783f7825a620c95a7b9dca919db790b879f/src/google/adk/telemetry/tracing.py#L650
    wrapped_functions = [
        Models.generate_content,
        Models.generate_content_stream,
        AsyncModels.generate_content,
        AsyncModels.generate_content_stream,
        Models.embed_content,
        AsyncModels.embed_content,
        InteractionsResource.create,
        AsyncInteractionsResource.create,
    ]

    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ):
        for func in wrapped_functions:
            co_filename = func.__code__.co_filename.replace("\\", "/")
            assert (
                "opentelemetry/instrumentation/google_genai" in co_filename
            ), (
                f"Expected opentelemetry/instrumentation/google_genai in {co_filename}"
            )

    for func in wrapped_functions:
        co_filename = func.__code__.co_filename.replace("\\", "/")
        assert (
            "opentelemetry/instrumentation/google_genai" not in co_filename
        ), (
            f"Expected opentelemetry/instrumentation/google_genai removed from {co_filename} upon uninstrument"
        )
