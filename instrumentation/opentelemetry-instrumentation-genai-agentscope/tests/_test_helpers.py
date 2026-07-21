# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Test utility functions."""

from __future__ import annotations

from typing import List

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)


def find_spans_by_name_prefix(
    spans: List[ReadableSpan], prefix: str
) -> List[ReadableSpan]:
    """Find spans by name prefix."""
    return [span for span in spans if span.name.startswith(prefix)]


def find_spans_by_operation(
    spans: List[ReadableSpan], operation_name: str
) -> List[ReadableSpan]:
    """Find spans by the ``gen_ai.operation.name`` attribute."""
    return [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)
        == operation_name
    ]


def print_span_tree(spans: List[ReadableSpan], indent: int = 0) -> None:
    """Print span tree structure for debugging."""
    sorted_spans = sorted(spans, key=lambda s: s.start_time)

    for span in sorted_spans:
        print("  " * indent + f"- {span.name}")
        print(
            "  " * indent
            + f"  Operation: {span.attributes.get(GenAIAttributes.GEN_AI_OPERATION_NAME)}"
        )
        print(
            "  " * indent
            + f"  Model: {span.attributes.get(GenAIAttributes.GEN_AI_REQUEST_MODEL)}"
        )

        child_spans = [
            s
            for s in spans
            if getattr(s, "parent", None)
            and s.parent
            and s.parent.span_id == span.context.span_id
        ]
        if child_spans:
            print_span_tree(child_spans, indent + 1)


def assert_no_removed_telemetry(spans: List[ReadableSpan]) -> None:
    """Assert that removed span kinds/attributes never appear.

    The migrated instrumentation drops the ``gen_ai.span.kind`` attribute
    (keeping only the standard ``gen_ai.operation.name``) and no longer emits
    ``react step`` / entry spans.
    """
    for span in spans:
        assert "gen_ai.span.kind" not in span.attributes, (
            f"gen_ai.span.kind must not be set (span={span.name})"
        )
        assert span.name != "react step", (
            "react step spans must not be emitted"
        )
        assert span.attributes.get("gen_ai.operation.name") != "react", (
            "react operation must not be emitted"
        )
