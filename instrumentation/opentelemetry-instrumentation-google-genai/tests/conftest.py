# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests configuration module."""

import pytest

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(scope="module")
def vcr_config():
    from opentelemetry.test_util_genai.vcr import (  # noqa: PLC0415
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
