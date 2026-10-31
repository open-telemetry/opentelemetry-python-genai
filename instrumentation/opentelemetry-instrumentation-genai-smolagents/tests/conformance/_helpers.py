# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for smolagents conformance scenarios."""

from __future__ import annotations

from typing import Any


def attr(span: dict[str, Any], name: str) -> Any:
    for entry in span["attributes"]:
        if entry["name"] == name:
            return entry["value"]
    return None


def chat_spans(report: Any) -> list[dict[str, Any]]:
    return [
        entry["span"]
        for entry in report["samples"]
        if "span" in entry
        and attr(entry["span"], "gen_ai.operation.name") == "chat"
    ]


def part_fields(
    messages: list[dict[str, Any]] | None,
) -> list[tuple[str, str | None]]:
    return [
        (part["type"], part.get("modality"))
        for message in (messages or [])
        for part in message["parts"]
    ]
