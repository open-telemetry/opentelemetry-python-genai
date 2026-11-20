# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the LangChain instrumentor lifecycle."""

from __future__ import annotations

from opentelemetry.instrumentation.genai.langchain import (
    LangChainInstrumentor,
)


def test_instrumentation_dependencies() -> None:
    """Test that instrumentation dependencies are correctly reported."""
    instrumentor = LangChainInstrumentor()
    dependencies = instrumentor.instrumentation_dependencies()

    assert dependencies == ("langchain-core >= 0.3.49, < 2",)
