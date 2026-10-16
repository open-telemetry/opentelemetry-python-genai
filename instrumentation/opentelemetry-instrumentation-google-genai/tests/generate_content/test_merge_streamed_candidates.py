# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    Part,
)

from opentelemetry.instrumentation.google_genai.generate_content import (
    _merge_streamed_candidates,
)
from opentelemetry.instrumentation.google_genai.message import (
    to_output_messages,
)


def _candidate(*parts, index=None, finish_reason=None):
    return Candidate(
        index=index,
        content=Content(role="model", parts=list(parts)),
        finish_reason=finish_reason,
    )


def test_merges_fragments_per_candidate_index():
    call = Part(
        function_call=FunctionCall(name="get_weather", args={"city": "Paris"})
    )
    merged = _merge_streamed_candidates(
        [
            _candidate(Part(text="Let me ", thought=True)),
            _candidate(Part(text="think.", thought=True)),
            _candidate(Part(text="It is ")),
            _candidate(Part(text="sunny."), index=1),
            _candidate(Part(text="warm.")),
            _candidate(call, finish_reason=FinishReason.STOP),
            _candidate(index=1, finish_reason=FinishReason.MAX_TOKENS),
        ]
    )

    assert merged == [
        Candidate(
            content=Content(
                role="model",
                parts=[
                    Part(text="Let me think.", thought=True),
                    Part(text="It is warm."),
                    call,
                ],
            ),
            finish_reason=FinishReason.STOP,
        ),
        Candidate(
            index=1,
            content=Content(role="model", parts=[Part(text="sunny.")]),
            finish_reason=FinishReason.MAX_TOKENS,
        ),
    ]


def test_does_not_mutate_the_streamed_chunks():
    first = _candidate(Part(text="Hello"))
    _merge_streamed_candidates([first, _candidate(Part(text=", world"))])

    assert first.content.parts == [Part(text="Hello")]


def test_contentless_candidate_produces_no_output_message():
    merged = _merge_streamed_candidates(
        [
            Candidate(index=0),
            Candidate(index=0, finish_reason=FinishReason.SAFETY),
        ]
    )

    assert merged == [Candidate(index=0, finish_reason=FinishReason.SAFETY)]
    assert to_output_messages(candidates=merged) == []
