# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for resolving error.type from google.genai exceptions."""

from __future__ import annotations

import httpx
import pytest

from opentelemetry.instrumentation.google_genai._error_type import (
    resolve_error_type,
)

try:
    # Google GenAI < 2.9.0
    from google.genai._interactions._exceptions import NotFoundError
except ImportError:
    try:
        # Google GenAI >= 2.9.0
        from google.genai._gaos.lib.compat_errors import NotFoundError
    except ImportError:
        NotFoundError = None


def test_non_api_exception_falls_back_to_the_class_name():
    assert resolve_error_type(ValueError("nope")) is None


@pytest.mark.skipif(
    NotFoundError is None,
    reason="Interactions are not supported in this version of google-genai",
)
def test_interactions_error_reports_its_status_code():
    # The interactions API raises from its own error hierarchy, unrelated to
    # google.genai.errors.APIError, carrying the status as `status_code`.
    response = httpx.Response(
        404,
        json={"error": {"message": "Interaction not found.", "code": 404}},
        request=httpx.Request("GET", "https://example.invalid/interactions/x"),
    )
    try:
        raise NotFoundError("Error code: 404", response=response, body=None)
    except Exception as exc:
        assert resolve_error_type(exc) == "404"
