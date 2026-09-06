# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for content-part conversion corner cases.

The realistic coverage lives in ``test_chat_completions.py``
(``test_chat_completion_multiturn_content_parts``,
``test_chat_completion_multimodal_content_parts`` and
``test_chat_completion_refusal``), which drive the instrumentation itself.
This module pins the shapes that are awkward to express as a recorded
request: a second audio format, malformed payloads, and content values a
well-formed request would not carry.
"""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.instrumentation.genai.openai.utils import (
    _prepare_input_messages,
)
from opentelemetry.util.genai.types import BlobPart, TextPart


def _parts(content):
    return _prepare_input_messages([{"role": "user", "content": content}])[
        0
    ].parts


def test_mp3_audio_uses_the_registered_mpeg_mime_type():
    # OpenAI's format is "mp3"; "audio/mp3" is not a registered media type.
    parts = _parts(
        [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": "ZmFrZSBtcDMgYnl0ZXM=",
                    "format": "mp3",
                },
            }
        ]
    )

    assert parts == [
        BlobPart(
            mime_type="audio/mpeg",
            modality="audio",
            content=b"fake mp3 bytes",
        )
    ]


def test_malformed_base64_audio_is_skipped():
    # Without strict validation this payload decodes to 3 junk bytes.
    parts = _parts(
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "$$$$abcd!!!!", "format": "wav"},
            },
            {"type": "text", "text": "still captured"},
        ]
    )

    assert parts == [TextPart(content="still captured")]


def test_file_without_id_or_data_is_skipped():
    parts = _parts(
        [
            {"type": "file", "file": {"filename": "spec.pdf"}},
            {"type": "text", "text": "still captured"},
        ]
    )

    assert parts == [TextPart(content="still captured")]


def test_unrecognized_part_costs_only_that_part():
    parts = _parts(
        [
            {"type": "someday_a_new_modality", "payload": "x"},
            {"type": "text", "text": "still captured"},
        ]
    )

    assert parts == [TextPart(content="still captured")]


def test_part_as_attribute_object():
    # Typed part objects resolve through get_property_value like dicts do.
    parts = _parts([SimpleNamespace(type="text", text="from an object")])

    assert parts == [TextPart(content="from an object")]


def test_mapping_content_yields_no_parts():
    # A bare dict is not a valid content shape; iterating it would have
    # produced a text part per KEY.
    parts = _parts({"type": "text", "text": "hi"})

    assert parts == []


def test_list_of_plain_strings():
    parts = _parts(["one", "two"])

    assert parts == [TextPart(content="one"), TextPart(content="two")]


def test_none_content_yields_no_parts():
    assert _parts(None) == []
