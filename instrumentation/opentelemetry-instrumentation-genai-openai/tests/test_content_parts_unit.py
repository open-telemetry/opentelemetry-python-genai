# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the content-part shapes that are awkward to record.

The recorded coverage lives in ``test_chat_completions.py``
(``test_chat_completion_captures_audio_and_file_input`` and
``test_chat_completion_captures_refusal_output``). This module pins the
second audio format, the Responses API file shape, and the malformed
descriptors a well-formed request would not carry.
"""

from __future__ import annotations

from opentelemetry.instrumentation.genai.openai.utils import (
    _content_to_parts,
)
from opentelemetry.util.genai.types import BlobPart, FilePart, TextPart


def test_mp3_audio_uses_the_registered_mpeg_media_type():
    # OpenAI's format is "mp3"; "audio/mp3" is not a registered type.
    parts = _content_to_parts(
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


def test_audio_of_unknown_format_has_no_media_type():
    parts = _content_to_parts(
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "aGVsbG8=", "format": "flac"},
            }
        ]
    )

    assert parts == [
        BlobPart(mime_type=None, modality="audio", content=b"hello")
    ]


def test_malformed_base64_audio_is_skipped():
    # Without strict validation this payload decodes to 3 junk bytes.
    parts = _content_to_parts(
        [
            {
                "type": "input_audio",
                "input_audio": {"data": "$$$$abcd!!!!", "format": "wav"},
            },
            {"type": "text", "text": "still captured"},
        ]
    )

    assert parts == [TextPart(content="still captured")]


def test_responses_input_file_by_id():
    parts = _content_to_parts([{"type": "input_file", "file_id": "file-abc"}])

    assert parts == [
        FilePart(mime_type=None, modality="document", file_id="file-abc")
    ]


def test_responses_input_file_inline_data():
    parts = _content_to_parts(
        [
            {
                "type": "input_file",
                "filename": "spec.pdf",
                "file_data": "data:application/pdf;base64,JVBERi0xLjQK",
            }
        ]
    )

    assert parts == [
        BlobPart(
            mime_type="application/pdf",
            modality="document",
            content=b"%PDF-1.4\n",
        )
    ]


def test_file_without_id_or_data_is_skipped():
    parts = _content_to_parts(
        [
            {"type": "file", "file": {"filename": "spec.pdf"}},
            {"type": "text", "text": "still captured"},
        ]
    )

    assert parts == [TextPart(content="still captured")]


def test_refusal_part_is_recorded_as_text():
    parts = _content_to_parts(
        [{"type": "refusal", "refusal": "I cannot help with that."}]
    )

    assert parts == [TextPart(content="I cannot help with that.")]
