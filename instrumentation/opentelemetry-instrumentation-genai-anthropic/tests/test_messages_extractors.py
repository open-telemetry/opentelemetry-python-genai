# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for Anthropic message parameter extraction."""

from io import BytesIO
from pathlib import Path

import pytest

from opentelemetry.instrumentation.genai.anthropic.messages_extractors import (
    extract_params,
    get_input_messages,
)
from opentelemetry.instrumentation.genai.anthropic.utils import (
    _convert_dict_block_to_part,
    convert_content_to_parts,
)
from opentelemetry.util.genai.types import (
    BlobPart,
    FilePart,
    GenericPart,
    TextPart,
    UriPart,
)


def test_extract_params_reads_sampling_params_from_extra_body():
    params = extract_params(
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40}
    )

    assert params.temperature == 0.7
    assert params.top_p == 0.9
    assert params.top_k == 40


def test_extract_params_prefers_named_sampling_params():
    params = extract_params(
        temperature=0.2,
        top_p=0.3,
        top_k=10,
        extra_body={"temperature": 0.7, "top_p": 0.9, "top_k": 40},
    )

    assert params.temperature == 0.2
    assert params.top_p == 0.3
    assert params.top_k == 10


def test_extract_params_ignores_non_numeric_extra_body_sampling_params():
    params = extract_params(
        extra_body={"temperature": "high", "top_p": True, "top_k": 2.5}
    )

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_extract_params_ignores_non_mapping_extra_body():
    params = extract_params(extra_body="temperature=0.7")

    assert params.temperature is None
    assert params.top_p is None
    assert params.top_k is None


def test_base64_image_source_converts_to_blob_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "QUJD",
            },
        }
    )

    assert isinstance(part, BlobPart)
    assert part.content == b"ABC"
    assert part.mime_type == "image/png"
    assert part.modality == "image"


def test_url_image_source_converts_to_uri_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {
                "type": "url",
                "url": "https://example.com/image.png",
            },
        }
    )

    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/image.png"
    assert part.mime_type is None
    assert part.modality == "image"


def test_file_image_source_converts_to_file_part():
    part = _convert_dict_block_to_part(
        {
            "type": "image",
            "source": {"type": "file", "file_id": "file-image"},
        }
    )

    assert isinstance(part, FilePart)
    assert part.file_id == "file-image"
    assert part.mime_type is None
    assert part.modality == "image"


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "image/png", "data": "%%%"},
        {"type": "base64", "media_type": "image/png"},
        {"type": "url"},
        {"type": "url", "url": ""},
        {"type": "file"},
        {"type": "file", "file_id": ""},
        {"type": "unknown", "data": "QUJD"},
        None,
    ],
)
def test_invalid_image_source_is_ignored(source):
    assert (
        _convert_dict_block_to_part({"type": "image", "source": source})
        is None
    )


def test_mixed_text_and_image_parts_preserve_order():
    parts = convert_content_to_parts(
        [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image",
                "source": {
                    "type": "url",
                    "url": "https://example.com/image.png",
                },
            },
        ]
    )

    assert len(parts) == 2
    assert isinstance(parts[0], TextPart)
    assert parts[0].content == "Describe this image."
    assert isinstance(parts[1], UriPart)
    assert parts[1].uri == "https://example.com/image.png"


def test_image_only_input_message_is_preserved():
    messages = get_input_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "QUJD",
                        },
                    }
                ],
            }
        ]
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert len(messages[0].parts) == 1
    assert isinstance(messages[0].parts[0], BlobPart)


def test_base64_document_source_converts_to_blob_part():
    parts = convert_content_to_parts(
        [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": "QUJD",
                },
            }
        ]
    )

    assert len(parts) == 1
    part = parts[0]
    assert isinstance(part, BlobPart)
    assert part.content == b"ABC"
    assert part.mime_type == "application/pdf"
    assert part.modality == "document"


def test_url_document_source_converts_to_uri_part():
    parts = convert_content_to_parts(
        [
            {
                "type": "document",
                "source": {
                    "type": "url",
                    "url": "https://example.com/document.pdf",
                },
            }
        ]
    )

    assert len(parts) == 1
    part = parts[0]
    assert isinstance(part, UriPart)
    assert part.uri == "https://example.com/document.pdf"
    assert part.mime_type == "application/pdf"
    assert part.modality == "document"


def test_file_document_source_converts_to_file_part():
    parts = convert_content_to_parts(
        [
            {
                "type": "document",
                "source": {"type": "file", "file_id": "file-document"},
            }
        ]
    )

    assert len(parts) == 1
    part = parts[0]
    assert isinstance(part, FilePart)
    assert part.file_id == "file-document"
    assert part.mime_type is None
    assert part.modality == "document"


def test_plain_text_document_source_converts_to_blob_part():
    parts = convert_content_to_parts(
        [
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": "Document text",
                },
            }
        ]
    )

    assert len(parts) == 1
    part = parts[0]
    assert isinstance(part, BlobPart)
    assert part.content == b"Document text"
    assert part.mime_type == "text/plain"
    assert part.modality == "document"


def test_nested_content_document_source_preserves_part_order():
    parts = convert_content_to_parts(
        [
            {
                "type": "document",
                "title": "Reference",
                "context": "Use the nested content.",
                "citations": {"enabled": True},
                "source": {
                    "type": "content",
                    "content": [
                        {"type": "text", "text": "Nested text"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.png",
                            },
                        },
                    ],
                },
            }
        ]
    )

    assert len(parts) == 1
    assert isinstance(parts[0], GenericPart)
    assert parts[0].type == "document"


@pytest.mark.parametrize(
    ("block_type", "media_type", "data"),
    [
        ("image", "image/png", Path("private/image.png")),
        ("image", "image/png", BytesIO(b"image")),
        (
            "document",
            "application/pdf",
            Path("private/document.pdf"),
        ),
        ("document", "application/pdf", BytesIO(b"document")),
    ],
)
def test_file_backed_base64_source_is_preserved_without_reading(
    block_type, media_type, data
):
    initial_position = data.tell() if isinstance(data, BytesIO) else None
    part = _convert_dict_block_to_part(
        {
            "type": block_type,
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }
    )

    assert isinstance(part, GenericPart)
    assert part.type == block_type
    if initial_position is not None:
        assert data.tell() == initial_position


@pytest.mark.parametrize(
    "source",
    [
        {"type": "base64", "media_type": "application/pdf", "data": "%%%"},
        {"type": "url"},
        {"type": "text"},
        {"type": "content", "content": None},
        {"type": "file"},
        {"type": "file", "file_id": ""},
        {"type": "unknown"},
        None,
    ],
)
def test_invalid_document_source_is_ignored(source):
    assert (
        convert_content_to_parts([{"type": "document", "source": source}])
        == []
    )
