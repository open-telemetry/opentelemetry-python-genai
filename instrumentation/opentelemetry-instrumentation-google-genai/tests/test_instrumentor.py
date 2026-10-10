# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GoogleGenAiSdkInstrumentor."""

from google.genai.models import AsyncModels, Models

from opentelemetry.instrumentation.google_genai import (
    GoogleGenAiSdkInstrumentor,
)
from opentelemetry.instrumentation.google_genai.interactions import (
    _HAS_INTERACTIONS,
    AsyncInteractionsResource,
    InteractionsResource,
)
from opentelemetry.test_util_genai.instrumentor import instrument


def _get_wrapped_functions():
    wrapped_functions = [
        Models.generate_content,
        Models.generate_content_stream,
        AsyncModels.generate_content,
        AsyncModels.generate_content_stream,
        Models.embed_content,
        AsyncModels.embed_content,
    ]
    # The interactions API only exists on newer google-genai versions; the
    # instrumentation skips wrapping it otherwise, so only assert on it there.
    if _HAS_INTERACTIONS:
        wrapped_functions += [
            InteractionsResource.create,
            AsyncInteractionsResource.create,
            InteractionsResource.get,
            AsyncInteractionsResource.get,
        ]
    return wrapped_functions


def test_co_filename_on_wrapped_functions(
    tracer_provider, logger_provider, meter_provider
):
    # ADK is relying on the __code__ attribute to suppress their instrumentation:
    # https://github.com/google/adk-python/blob/0d4d3783f7825a620c95a7b9dca919db790b879f/src/google/adk/telemetry/tracing.py#L650
    wrapped_functions = _get_wrapped_functions()

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


def test_uninstrument_after_reinstantiation(
    tracer_provider, logger_provider, meter_provider
):
    wrapped_functions = _get_wrapped_functions()

    with instrument(
        GoogleGenAiSdkInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
    ) as inst:
        assert inst.is_instrumented_by_opentelemetry is True

        # Re-instantiating the singleton instrumentor runs __init__, which must
        # not wipe saved snapshots (generate_content, interactions, embeddings)
        # needed for uninstrumentation.
        reinstantiated = GoogleGenAiSdkInstrumentor()
        assert reinstantiated.is_instrumented_by_opentelemetry is True

    # After the context manager exits, uninstrument() was called on the
    # re-instantiated singleton. Verify uninstrumented state.
    assert reinstantiated.is_instrumented_by_opentelemetry is False
    assert inst.is_instrumented_by_opentelemetry is False

    # Confirm all wrapped functions across generate-content, embeddings,
    # and interactions are restored.
    for func in wrapped_functions:
        co_filename = func.__code__.co_filename.replace("\\", "/")
        assert (
            "opentelemetry/instrumentation/google_genai" not in co_filename
        ), (
            "Expected opentelemetry/instrumentation/google_genai "
            f"removed from {co_filename} upon uninstrument"
        )
