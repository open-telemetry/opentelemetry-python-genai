# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval execution for DSPy."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import dspy
from dspy.dsp.utils import dotdict


def fake_get_request(url: str, query: str, k: int) -> list[Any]:
    return [
        dotdict(
            {
                "long_text": f"ColBERT result {i} for {query}",
                "pid": 100 + i,
                "score": round(0.95 - i * 0.1, 2),
            }
        )
        for i in range(k)
    ]


colbert = dspy.ColBERTv2(url="http://colbert.example.com:8893/api/search")
with patch(
    "dspy.dsp.colbertv2.colbertv2_get_request", side_effect=fake_get_request
):
    colbert("What is OpenTelemetry?", k=2)
