# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator

import pytest

pytest.register_assert_rewrite("opentelemetry.test_util_genai.vcr")

from opentelemetry.test_util_genai.vcr import (
    scrub_response_headers_overwrite,
)
from opentelemetry.util.genai.handler import get_telemetry_handler

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


@pytest.fixture
def reset_telemetry_handler() -> Iterator[None]:
    """Drop util-genai's process-wide ``TelemetryHandler`` around each test.

    ``get_telemetry_handler`` caches the first handler it builds. A test that
    instruments without explicit providers would otherwise pin every later
    test to the global ones, so their in-memory exporters would come back
    empty no matter what the instrumentation did.
    """

    def _clear() -> None:
        if (
            getattr(get_telemetry_handler, "_default_handler", None)
            is not None
        ):
            delattr(get_telemetry_handler, "_default_handler")

    _clear()
    yield
    _clear()
