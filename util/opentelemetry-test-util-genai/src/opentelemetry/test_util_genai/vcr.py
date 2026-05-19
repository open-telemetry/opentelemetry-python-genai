# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""VCR cassette helpers for GenAI instrumentation tests.

Every instrumentation conftest used to duplicate:

- A ``LiteralBlockScalar`` YAML representer that renders long strings and
  pretty-printed JSON bodies as literal block scalars (``|``-style) so
  cassettes are diff-readable instead of one-line escaped blobs.
- A ``PrettyPrintJSONBody`` VCR serializer that runs the representer over
  ``request.body`` and ``response.body.string`` fields.
- A ``fixture_vcr`` autouse fixture that registers the serializer with
  ``pytest-vcr``.
- A per-package ``scrub_response_headers`` callable.

This module hosts those bits once. Per-instrumentation conftests stay short:

.. code-block:: python

    from opentelemetry.test_util_genai.fixtures import *
    from opentelemetry.test_util_genai.vcr import (
        scrub_response_headers,
    )

    @pytest.fixture(scope="module")
    def vcr_config():
        return {
            "filter_headers": [
                ("authorization", "Bearer test_openai_api_key"),
                ("openai-organization", "test_openai_org_id"),
            ],
            "decode_compressed_response": True,
            "before_record_response": scrub_response_headers(
                ["openai-organization", "set-cookie"]
            ),
        }
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from typing import Any

import pytest
import yaml


class LiteralBlockScalar(str):
    """A string subclass that renders as a YAML literal block scalar.

    The custom representer below emits these as ``|``-style blocks, which
    preserves whitespace and avoids the escape-soup that one-line
    JSON-in-YAML produces.
    """


def _literal_block_scalar_presenter(
    dumper: yaml.Dumper, data: LiteralBlockScalar
) -> yaml.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


yaml.add_representer(LiteralBlockScalar, _literal_block_scalar_presenter)


def _process_string_value(string_value: str) -> str:
    """Pretty-print as JSON if it parses, else wrap long strings as a block."""
    try:
        json_data = json.loads(string_value)
    except (ValueError, TypeError):
        if len(string_value) > 80:
            return LiteralBlockScalar(string_value)
        return string_value
    return LiteralBlockScalar(json.dumps(json_data, indent=2))


def _convert_body_to_literal(data: Any) -> Any:
    """Recursively pretty-print every ``body`` field in a cassette dict."""
    if isinstance(data, dict):
        for key, value in data.items():
            # Response body case: response.body.string
            if (
                key == "body"
                and isinstance(value, dict)
                and "string" in value
                and isinstance(value["string"], str)
            ):
                value["string"] = _process_string_value(value["string"])
            # Request body case: request.body
            elif key == "body" and isinstance(value, str):
                data[key] = _process_string_value(value)
            else:
                _convert_body_to_literal(value)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            data[idx] = _convert_body_to_literal(item)
    return data


class PrettyPrintJSONBody:
    """A VCR cassette serializer that pretty-prints JSON request/response bodies.

    Register via ``vcr.register_serializer("yaml", PrettyPrintJSONBody)`` —
    the ``fixture_vcr`` fixture in this module does that automatically.
    """

    @staticmethod
    def serialize(cassette_dict: dict[str, Any]) -> str:
        cassette_dict = _convert_body_to_literal(cassette_dict)
        return yaml.dump(
            cassette_dict, default_flow_style=False, allow_unicode=True
        )

    @staticmethod
    def deserialize(cassette_string: str) -> dict[str, Any]:
        return yaml.load(cassette_string, Loader=yaml.Loader)


@pytest.fixture(scope="module", autouse=True)
def fixture_vcr(vcr: Any) -> Any:
    """Autouse fixture registering ``PrettyPrintJSONBody`` with ``pytest-vcr``.

    Imported via ``from opentelemetry.test_util_genai.vcr import fixture_vcr``
    in a per-package conftest. The ``vcr`` parameter is supplied by
    ``pytest-vcr``.
    """
    vcr.register_serializer("yaml", PrettyPrintJSONBody)
    return vcr


def scrub_response_headers(
    headers_to_scrub: Iterable[str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a ``before_record_response`` callable that drops named headers.

    Header matching is case-insensitive. Use as::

        "before_record_response": scrub_response_headers([
            "openai-organization",
            "set-cookie",
        ])
    """
    targets = {h.lower() for h in headers_to_scrub}

    def _scrub(response: dict[str, Any]) -> dict[str, Any]:
        headers = response.get("headers")
        if isinstance(headers, dict):
            for key in list(headers):
                if key.lower() in targets:
                    headers.pop(key, None)
        return response

    return _scrub


def scrub_response_headers_overwrite(
    replacements: dict[str, str],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a ``before_record_response`` callable that overwrites named headers.

    Use when the recorded cassette must keep the header surface but with
    deterministic test values (e.g. so playback matches a client that still
    reads the header). Header matching is case-insensitive on lookup; the
    stored key is preserved if present, otherwise the key from
    ``replacements`` is used.

    Use as::

        "before_record_response": scrub_response_headers_overwrite({
            "openai-organization": "test_openai_org_id",
            "Set-Cookie": "test_set_cookie",
        })
    """
    targets = {k.lower(): (k, v) for k, v in replacements.items()}

    def _scrub(response: dict[str, Any]) -> dict[str, Any]:
        headers = response.get("headers")
        if not isinstance(headers, dict):
            return response
        existing_by_lower = {k.lower(): k for k in headers}
        for lower, (default_key, value) in targets.items():
            key = existing_by_lower.get(lower, default_key)
            headers[key] = value
        return response

    return _scrub
