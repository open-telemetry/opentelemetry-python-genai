# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance scenario: retrieval execution for DSPy."""

from __future__ import annotations

from typing import Any

import dspy


class _DummyPassage:
    def __init__(self, text: str, pid: str, score: float) -> None:
        self.long_text = text
        self.pid = pid
        self.score = score


class _DummyRM:
    def __call__(self, query: str, k: int = 3, **kwargs: Any) -> list[Any]:
        return [
            _DummyPassage(
                f"Passage {i} for {query}",
                pid=f"doc-{i}",
                score=round(0.95 - i * 0.1, 2),
            )
            for i in range(k)
        ]


dspy.settings.configure(rm=_DummyRM())
retrieve = dspy.Retrieve(k=2)
retrieve("What is OpenTelemetry?")
