# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across langchain conformance scenarios."""

from __future__ import annotations

from typing import Any

from opentelemetry.test.weaver_live_check import LiveCheckReport


def span_attribute_values(report: LiveCheckReport, name: str) -> list[Any]:
    """Return every value of ``name`` across all span samples in ``report``."""
    return [
        attr["value"]
        for entry in report["samples"]
        if "span" in entry
        for attr in entry["span"]["attributes"]
        if attr["name"] == name
    ]
