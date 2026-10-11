# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Content assertions on reports from the shared conformance runner."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import pytest
from genai_conformance import genai_session

from opentelemetry.conformance import ConformanceSession, require_pin

if TYPE_CHECKING:
    from opentelemetry.conformance import ScenarioReport


def messages(
    result: ScenarioReport, attribute: str
) -> Iterator[dict[str, Any]]:
    assert result.report is not None, "scenario produced no Weaver report"
    for sample in result.report["samples"]:
        span = sample.get("span")
        if span is None:
            continue
        for attr in span["attributes"]:
            if attr["name"] == attribute:
                yield from json.loads(attr["value"])


def assert_message_part(
    result: ScenarioReport,
    attribute: str,
    part_type: str,
    modality: str | None = None,
) -> None:
    parts = [
        part
        for message in messages(result, attribute)
        for part in message["parts"]
    ]
    assert any(
        part["type"] == part_type
        and (modality is None or part.get("modality") == modality)
        for part in parts
    ), f"{result.name}: missing {part_type}/{modality} in {attribute}: {parts}"


@pytest.fixture(scope="module")
def common_session(
    request: pytest.FixtureRequest,
) -> Iterator[ConformanceSession]:
    directory = request.path.parent / "conformance"
    ref = require_pin(
        directory.parents[3] / "versions.env", "SEMCONV_GENAI_REF"
    )
    with genai_session(
        directory, variables={"SEMCONV_GENAI_REF": ref}
    ) as session:
        yield session
