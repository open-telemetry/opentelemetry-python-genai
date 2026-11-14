# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests configuration module."""

import os

import pytest

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch) -> None:
    if not os.getenv("GOOGLE_GENAI_API_VERSION"):
        # google-genai >= 3.0.0 defaults the Gemini Developer API to v1;
        # VCR cassettes are recorded against v1beta (the < 3.0.0 default).
        monkeypatch.setenv("GOOGLE_GENAI_API_VERSION", "beta")


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (
        scrub_response_headers_overwrite,
    )

    return {
        "filter_headers": [
            ("authorization", "Bearer test_google_genai_api_key"),
            ("x-goog-api-key", "test_google_genai_api_key"),
            ("x-goog-user-project", "test_project"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "x-goog-api-key": "test_google_genai_api_key",
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }
