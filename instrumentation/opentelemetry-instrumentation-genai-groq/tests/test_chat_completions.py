# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_OPERATION_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GenAiOperationNameValues,
    GenAiSystemValues,
)
from groq import Groq


@pytest.mark.vcr
def test_chat_completions_basic(
    instrument_no_content,
    span_exporter,
    vcr,
    groq_client: Groq,
):
    with vcr.use_cassette("test_chat_completions_basic.yaml"):
        response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "user", "content": "Tell me a joke"},
        ],
    )

    assert response is not None
    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    assert span.attributes.get(GEN_AI_SYSTEM) == "groq" or span.attributes.get("gen_ai.provider.name") == "groq"
    assert span.attributes.get(GEN_AI_OPERATION_NAME) == GenAiOperationNameValues.CHAT.value
    assert span.attributes[GEN_AI_REQUEST_MODEL] == "llama3-8b-8192"
    assert "gen_ai.response.model" in span.attributes
