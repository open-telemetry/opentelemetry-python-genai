# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test configuration and fixtures for Sentence Transformers instrumentation."""
# pylint: disable=redefined-outer-name

import pytest

from opentelemetry.instrumentation.genai.sentence_transformers import (
    SentenceTransformersInstrumentor,
)
from opentelemetry.test_util_genai.instrumentor import instrument

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
]


@pytest.fixture
def instrument_no_content(tracer_provider, logger_provider, meter_provider):
    """Instrument Sentence Transformers without content capture."""
    with instrument(
        SentenceTransformersInstrumentor(),
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        content_capture="NO_CONTENT",
    ) as instrumentor:
        yield instrumentor
