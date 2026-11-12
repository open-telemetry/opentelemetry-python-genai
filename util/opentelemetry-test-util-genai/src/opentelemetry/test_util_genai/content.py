# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema

_SCHEMAS_DIR = Path(__file__).resolve().parent / "_schemas"

SCHEMA_FILES: dict[str, str] = {
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.definitions": "gen-ai-tool-definitions.json",
    "gen_ai.retrieval.documents": "gen-ai-retrieval-documents.json",
}


@functools.cache
def _validator(name: str) -> jsonschema.Draft202012Validator:
    path = _SCHEMAS_DIR / SCHEMA_FILES[name]
    return jsonschema.Draft202012Validator(
        json.loads(path.read_text(encoding="utf-8"))
    )


def _to_json_like(value: Any, path: str) -> Any:
    # SDK attributes store sequences as tuples; anything else (list, JSON string,
    # dataclass) means the value wasn't converted to AnyValue.
    if isinstance(value, tuple):
        return [_to_json_like(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        return {k: _to_json_like(v, f"{path}.{k}") for k, v in value.items()}
    assert value is None or isinstance(
        value, (str, bool, int, float, bytes)
    ), f"{path}: unexpected {type(value).__name__}"
    return value


def validate_content_attribute(name: str, value: Any) -> Any:
    """Validate and return a structured GenAI content attribute as plain lists/dicts.

    Asserts the value is a structured sequence and matches the semconv JSON schema.
    """
    assert isinstance(value, tuple), (
        f"{name}: expected tuple, got {type(value).__name__}"
    )
    normalized = _to_json_like(value, name)
    errors = [
        e.message
        for e in sorted(
            _validator(name).iter_errors(normalized),
            key=lambda err: str(err.path),
        )
    ]
    assert not errors, f"{name} does not match the semconv schema: {errors}"
    return normalized


def get_content_attribute(
    attributes: Mapping[str, Any], name: str
) -> list[dict[str, Any]]:
    """Return a structured GenAI content attribute as plain lists/dicts.

    Asserts the value is a structured sequence and matches the semconv JSON schema.
    """
    return validate_content_attribute(name, attributes[name])
