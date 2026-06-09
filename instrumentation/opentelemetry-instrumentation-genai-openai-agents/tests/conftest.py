# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from opentelemetry.test_util_genai.vcr import scrub_response_headers_overwrite

pytest_plugins = [
    "opentelemetry.test_util_genai.fixtures",
    "opentelemetry.test_util_genai.vcr",
]


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [
            ("cookie", "test_cookie"),
            ("authorization", "Bearer test_openai_api_key"),
            ("openai-organization", "test_openai_org_id"),
            ("openai-project", "test_openai_project_id"),
        ],
        "decode_compressed_response": True,
        "before_record_response": scrub_response_headers_overwrite(
            {
                "openai-organization": "test_openai_org_id",
                "openai-project": "test_openai_project_id",
                "Set-Cookie": "test_set_cookie",
            }
        ),
    }
