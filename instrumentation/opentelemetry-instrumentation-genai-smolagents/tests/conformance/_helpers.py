# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for smolagents conformance scenarios."""

from __future__ import annotations

import json
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


def part_fields(messages_json: str | None) -> list[tuple[str, str | None]]:
    messages = json.loads(messages_json) if messages_json else []
    return [
        (part["type"], part.get("modality"))
        for message in messages
        for part in message["parts"]
    ]
